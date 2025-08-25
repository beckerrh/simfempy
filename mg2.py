# hierarchical_mlp_vcycle_with_testing.py
# Requires: jax, optax, numpy
# pip install --upgrade "jax[cpu]" optax

from typing import Sequence, Dict, Any, Tuple, Iterable, List
import jax
import jax.numpy as jnp
import optax
import numpy as np

PRNGKey = jax.random.PRNGKey

# -------------------------
# Basic model utilities (pure functional)
# -------------------------
def glorot_init(rng, shape):
    lim = jnp.sqrt(6.0 / (shape[0] + shape[1]))
    return jax.random.uniform(rng, shape, minval=-lim, maxval=lim)

def init_params(rng: PRNGKey, dims: Sequence[int]) -> Dict[str, Dict[str, jnp.ndarray]]:
    params = {}
    keys = jax.random.split(rng, len(dims)-1)
    for i in range(len(dims)-1):
        in_dim = dims[i]
        out_dim = dims[i+1]
        k = glorot_init(keys[i], (in_dim, out_dim))
        b = jnp.zeros((out_dim,))
        params[f"dense_{i}"] = {'kernel': k, 'bias': b}
    return params

def forward(params: Dict[str, Dict[str, jnp.ndarray]],
            x: jnp.ndarray,
            act=jax.nn.relu) -> jnp.ndarray:
    h = x
    n_layers = len(params)
    for i in range(n_layers-1):
        layer = params[f'dense_{i}']
        h = jnp.dot(h, layer['kernel']) + layer['bias']
        h = act(h)
    last = params[f'dense_{n_layers-1}']
    out = jnp.dot(h, last['kernel']) + last['bias']
    return out

# -------------------------
# Mapping helpers & initial mapping
# -------------------------
def init_mapping_for_dims(dims: Sequence[int]) -> Dict[str, List[List[int]]]:
    mapping = {}
    for i in range(len(dims)-1):
        out_dim = dims[i+1]
        mapping[f'dense_{i}'] = [[j] for j in range(out_dim)]
    return mapping

# -------------------------
# Split neurons (uses numpy internally for shape-changing ops)
# -------------------------
def split_neurons_in_params(params: Dict[str, Dict[str, np.ndarray]],
                            mapping: Dict[str, List[List[int]]],
                            layer_name: str,
                            neuron_indices: Sequence[int],
                            eps: float = 1e-3,
                            rng: PRNGKey = None
                            ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, List[List[int]]]]:
    """
    Splits specified output neurons (coarse indices) at `layer_name`.
    params expected as numpy arrays (we'll convert if given jax arrays).
    Returns (params_new, mapping_new) as numpy-backed dicts.
    """
    # ensure numpy arrays
    params_u = {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in params.items()}
    mapping_u = {k: [list(lst) for lst in v] for k, v in mapping.items()}

    if rng is None:
        rng = jax.random.PRNGKey(0)

    assert layer_name in params_u
    K = params_u[layer_name]['kernel']  # (in_dim, out_dim)
    b = params_u[layer_name]['bias']
    in_dim, out_dim = K.shape
    idx = np.array(neuron_indices, dtype=int)
    m = idx.size
    if m == 0:
        return params_u, mapping_u

    # random directions
    u = np.random.normal(size=(in_dim, m)).astype(np.float32)
    beta = np.random.normal(size=(m,)).astype(np.float32)

    K_cols = K[:, idx]            # (in_dim, m)
    Ka = K_cols + eps * u
    Kb = K_cols - eps * u
    K_new = np.concatenate([K, Ka, Kb], axis=1)   # (in_dim, out_dim + 2m)

    b_cols = b[idx]
    ba = b_cols + eps * beta
    bb = b_cols - eps * beta
    b_new = np.concatenate([b, ba, bb], axis=0)

    params_u[layer_name]['kernel'] = K_new
    params_u[layer_name]['bias'] = b_new

    # update mapping: appended children indices are at the end
    appended_start = out_dim
    for sel_i_pos, coarse_idx in enumerate(idx.tolist()):
        # append two children indices for this coarse index
        mapping_u[layer_name][coarse_idx].extend([appended_start + 2*sel_i_pos,
                                                  appended_start + 2*sel_i_pos + 1])

    # adjust next layer if present
    assert layer_name.startswith("dense_")
    cur_idx = int(layer_name.split("_")[1])
    next_layer_name = f'dense_{cur_idx+1}'
    if next_layer_name in params_u:
        K_next = params_u[next_layer_name]['kernel']  # (in_next, out_next)
        rows = K_next[idx, :]    # (m, out_next)
        Ka_rows = 0.5 * rows
        Kb_rows = 0.5 * rows
        K_next_new = np.concatenate([K_next, Ka_rows, Kb_rows], axis=0)
        params_u[next_layer_name]['kernel'] = K_next_new

    return params_u, mapping_u

# -------------------------
# Restrict (fine->coarse) & Prolongate (coarse->fine)
# -------------------------
def restrict_params_from_fine(params_fine: Dict[str, Dict[str, np.ndarray]],
                              mapping_coarse: Dict[str, List[List[int]]]
                              ) -> Dict[str, Dict[str, np.ndarray]]:
    params_coarse = {}
    layer_names = list(mapping_coarse.keys())
    for layer_name in layer_names:
        Kf = np.array(params_fine[layer_name]['kernel'])
        bf = np.array(params_fine[layer_name]['bias'])
        groups = mapping_coarse[layer_name]
        Kc_cols = []
        bc = []
        for g in groups:
            cols = Kf[:, g]   # (in_dim, len(g))
            avg_col = np.mean(cols, axis=1)
            Kc_cols.append(avg_col)
            bc.append(np.mean(bf[g]))
        Kc = np.stack(Kc_cols, axis=1)
        bc = np.array(bc)
        params_coarse[layer_name] = {'kernel': Kc, 'bias': bc}

    # compress next-layer rows
    for i in range(len(layer_names)-1):
        next_name = f'dense_{i+1}'
        if next_name in params_coarse:
            K_next_f = np.array(params_fine[next_name]['kernel'])  # (in_fine, out_next)
            groups = mapping_coarse[f'dense_{i}']
            rows_grouped = []
            for g in groups:
                rows = K_next_f[g, :]   # (len(g), out_next)
                avg_row = np.mean(rows, axis=0)
                rows_grouped.append(avg_row)
            K_next_c = np.stack(rows_grouped, axis=0)
            params_coarse[next_name]['kernel'] = K_next_c
    return params_coarse

def prolongate_params_from_coarse(params_coarse: Dict[str, Dict[str, np.ndarray]],
                                  mapping_coarse: Dict[str, List[List[int]]],
                                  fine_template_params: Dict[str, Dict[str, np.ndarray]]
                                  ) -> Dict[str, Dict[str, np.ndarray]]:
    params_fine = {}
    layer_names = list(mapping_coarse.keys())
    for layer_name in layer_names:
        fine_layer = fine_template_params[layer_name]
        Kf_new = np.zeros_like(fine_layer['kernel'])
        bf_new = np.zeros_like(fine_layer['bias'])
        groups = mapping_coarse[layer_name]
        Kc = np.array(params_coarse[layer_name]['kernel'])
        bc = np.array(params_coarse[layer_name]['bias'])
        for j, g in enumerate(groups):
            for fine_idx in g:
                Kf_new[:, fine_idx] = Kc[:, j]
                bf_new[fine_idx] = bc[j]
        params_fine[layer_name] = {'kernel': Kf_new, 'bias': bf_new}

    # next-layer rows copy:
    for i in range(len(layer_names)-1):
        next_name = f'dense_{i+1}'
        if next_name in params_fine:
            K_next_template = fine_template_params[next_name]['kernel']
            K_next_new = np.zeros_like(K_next_template)
            K_next_coarse = np.array(params_coarse[next_name]['kernel'])
            groups = mapping_coarse[f'dense_{i}']
            for coarse_row_idx, g in enumerate(groups):
                coarse_row = K_next_coarse[coarse_row_idx, :]
                for fine_row in g:
                    K_next_new[fine_row, :] = coarse_row
            params_fine[next_name]['kernel'] = K_next_new
    return params_fine

# -------------------------
# Loss, indicator, Dörfler mark
# -------------------------
def mse_loss(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((preds - targets)**2, axis=-1)

def neuron_gradient_indicator(params: Dict[str, Dict[str, Any]],
                              loss_fn,
                              batch: Tuple[jnp.ndarray, jnp.ndarray],
                              layer_name: str) -> np.ndarray:
    xb, yb = batch
    # Make sure params is jax-friendly (convert numpy -> jnp)
    params_j = {k: {'kernel': jnp.array(v['kernel']), 'bias': jnp.array(v['bias'])} for k, v in params.items()}
    def scalar_loss(p):
        preds = forward(p, xb)
        return jnp.sum(loss_fn(preds, yb))
    grads = jax.grad(scalar_loss)(params_j)
    g_layer = grads[layer_name]
    gK = g_layer['kernel']   # jnp
    gb = g_layer['bias']
    gK2 = jnp.sum(gK**2, axis=0)
    eta = jnp.sqrt(gK2 + gb**2)
    return np.array(eta)

def dorfler_marking(eta: np.ndarray, theta: float = 0.6) -> np.ndarray:
    vals = np.array(eta)
    if vals.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(vals)[::-1]
    sorted_vals = vals[order]
    cumsum = np.cumsum(sorted_vals**2)
    total = np.sum(sorted_vals**2)
    if total <= 0:
        return np.array([], dtype=int)
    k = np.searchsorted(cumsum, theta * total)
    k = min(k, vals.size - 1)
    selected = order[:k+1]
    return selected

# -------------------------
# Optax helpers (jax-jitted where appropriate)
# -------------------------
def create_optimizer(learning_rate: float = 1e-3):
    return optax.adam(learning_rate)

def init_opt_state(opt, params):
    # params may be numpy; convert to jax tree with jnp arrays
    params_j = {k: {'kernel': jnp.array(v['kernel']), 'bias': jnp.array(v['bias'])} for k, v in params.items()}
    return opt.init(params_j)

@jax.jit
def _compute_grads_jax(params_j, batch, loss_fn):
    return jax.grad(lambda p: jnp.mean(loss_fn(forward(p, batch[0]), batch[1])))(params_j)

def compute_grads(params: Dict[str, Dict[str, Any]], batch, loss_fn):
    # convert to jax arrays, compute grads, convert back to numpy arrays
    params_j = {k: {'kernel': jnp.array(v['kernel']), 'bias': jnp.array(v['bias'])} for k, v in params.items()}
    grads_j = _compute_grads_jax(params_j, batch, loss_fn)
    # convert grads to numpy for apply_updates via optax (we'll use optax.apply_updates on jax arrays though)
    return grads_j  # keep as jax pytree

def apply_updates_with_opt(opt, params, opt_state, grads):
    # params: numpy dict -> convert to jax tree
    params_j = {k: {'kernel': jnp.array(v['kernel']), 'bias': jnp.array(v['bias'])} for k, v in params.items()}
    updates, new_opt_state = opt.update(grads, opt_state, params_j)
    params_updated_j = optax.apply_updates(params_j, updates)
    # convert back to numpy dict (for consistent storage across code)
    params_updated = {k: {'kernel': np.array(v['kernel']), 'bias': np.array(v['bias'])} for k, v in params_updated_j.items()}
    return params_updated, new_opt_state

# -------------------------
# V-cycle (recursive, operates on levels list where params are numpy arrays)
# -------------------------
def v_cycle(levels: List[Dict[str, Any]],
            level_idx: int,
            train_batches: Iterable[Tuple[jnp.ndarray, jnp.ndarray]],
            Sc: int,
            Sf: int,
            loss_fn,
            learning_rate: float = 1e-3):
    level = levels[level_idx]
    params = level['params']      # numpy dict
    opt = create_optimizer(learning_rate)
    opt_state = init_opt_state(opt, params)
    train_iter = iter(train_batches)

    # pre-smooth
    for _ in range(Sc):
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        grads = compute_grads(params, batch, loss_fn)
        params, opt_state = apply_updates_with_opt(opt, params, opt_state, grads)

    # recurse to coarse if possible
    if level_idx > 0:
        coarse_level = levels[level_idx - 1]
        mapping_coarse = coarse_level['mapping']
        # Restrict fine->coarse
        params_coarse_init = restrict_params_from_fine(params, mapping_coarse)
        levels[level_idx - 1]['params'] = params_coarse_init
        # recursive call
        params_coarse_updated = v_cycle(levels, level_idx - 1, train_batches, Sc, Sf, loss_fn, learning_rate)
        # prolongate updated coarse -> fine
        fine_template = level['template']
        params_prolong = prolongate_params_from_coarse(params_coarse_updated, mapping_coarse, fine_template)
        # Here: additive correction pattern could be used; we simply replace for simplicity
        params = params_prolong

    # post-smooth
    opt_state = init_opt_state(opt, params)
    for _ in range(Sf):
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        grads = compute_grads(params, batch, loss_fn)
        params, opt_state = apply_updates_with_opt(opt, params, opt_state, grads)

    levels[level_idx]['params'] = params
    return params

# -------------------------
# Top-level training with testing & Dörfler marking integrated
# -------------------------
def hierarchical_training_with_vcycle_and_testing(
    rng: PRNGKey,
    initial_dims: Sequence[int],
    outer_iters: int,
    train_batches: Iterable[Tuple[jnp.ndarray, jnp.ndarray]],
    val_batch_for_estimate: Tuple[jnp.ndarray, jnp.ndarray],
    Sc: int = 100,
    Sf: int = 50,
    theta: float = 0.6,
    split_eps: float = 1e-3,
    loss_fn = mse_loss,
    learning_rate: float = 1e-3
):
    # initialize coarsest
    rng, sub = jax.random.split(rng)
    params0_j = init_params(sub, initial_dims)   # jnp arrays
    # convert to numpy storage for splitting convenience
    params0 = {k: {'kernel': np.array(v['kernel']), 'bias': np.array(v['bias'])} for k, v in params0_j.items()}
    mapping0 = init_mapping_for_dims(initial_dims)
    template0 = {k: {kk: np.array(vv) for kk, vv in layer.items()} for k, layer in params0.items()}

    levels = [{'params': params0, 'mapping': mapping0, 'template': template0}]

    train_iter = iter(train_batches)

    for outer in range(outer_iters):
        print(f"\n=== Outer iter {outer} | n_levels = {len(levels)} ===")
        finest_idx = len(levels) - 1
        # run V-cycle starting at finest
        v_cycle(levels, finest_idx, train_batches, Sc, Sf, loss_fn, learning_rate)

        # TEST: compute indicators on finest level
        finest = levels[-1]
        params_finest = finest['params']
        eta = neuron_gradient_indicator(params_finest, loss_fn, val_batch_for_estimate, layer_name='dense_0')
        marked = dorfler_marking(eta, theta=theta)
        print("Marked neurons (finest):", marked)

        if marked.size == 0:
            print("No marks -> stopping outer loop early.")
            break

        # REFINE: split marked coarse indices at finest to append a new finer level
        params_new, mapping_new = split_neurons_in_params(params_finest, finest['mapping'],
                                                         layer_name='dense_0', neuron_indices=marked, eps=split_eps, rng=rng)
        template_new = {k: {kk: np.array(vv) for kk, vv in layer.items()} for k, layer in params_new.items()}
        levels.append({'params': params_new, 'mapping': mapping_new, 'template': template_new})
        print("Appended new finer level; now levels =", len(levels))

    return levels

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import itertools

    # Toy data generator
    def batch_gen(batch_size=64):
        rng_np = np.random.RandomState(0)
        while True:
            xb = rng_np.randn(batch_size, 4).astype(np.float32)
            yb = (np.sum(xb**2, axis=1, keepdims=True)).astype(np.float32)
            yield (jnp.array(xb), jnp.array(yb))

    rng = jax.random.PRNGKey(0)
    initial_dims = [4, 32, 1]   # coarse net: 4 -> 32 -> 1
    train_batches = batch_gen()
    val_batch = next(train_batches)

    levels = hierarchical_training_with_vcycle_and_testing(
        rng,
        initial_dims=initial_dims,
        outer_iters=3,
        train_batches=train_batches,
        val_batch_for_estimate=val_batch,
        Sc=200,
        Sf=100,
        theta=0.6,
        split_eps=1e-3,
        loss_fn=mse_loss,
        learning_rate=1e-3
    )

    print("\nFinal hierarchy summary:")
    for i, lvl in enumerate(levels):
        print(f" Level {i}:")
        for k, v in lvl['params'].items():
            print(f"  {k}: kernel {np.array(v['kernel']).shape}, bias {np.array(v['bias']).shape}")