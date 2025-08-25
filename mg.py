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
    """
    dims: [d_in, w1, w2, ..., d_out]
    returns params dict: {'dense_0': {'kernel': (in, out), 'bias': (out,)}, ...}
    """
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
        layer = params[f"dense_{i}"]
        h = jnp.dot(h, layer['kernel']) + layer['bias']
        h = act(h)
    last = params[f"dense_{n_layers-1}"]
    out = jnp.dot(h, last['kernel']) + last['bias']
    return out

# -------------------------
# Mapping helpers & initial mapping
# -------------------------
def init_mapping_for_dims(dims: Sequence[int]) -> Dict[str, List[List[int]]]:
    """
    For an MLP with layers dims -> create initial identity mapping per layer.
    mapping['dense_i'] is a list of lists of fine indices for each coarse output.
    Initially each coarse output corresponds to exactly one fine index: [[0],[1],...].
    """
    mapping = {}
    for i in range(len(dims)-1):
        out_dim = dims[i+1]
        mapping[f"dense_{i}"] = [[j] for j in range(out_dim)]
    return mapping

# -------------------------
# Split neurons (fine-level creation) -> returns new params and mapping for new fine level
# -------------------------
def split_neurons_in_params(params: Dict[str, Dict[str, jnp.ndarray]],
                            mapping: Dict[str, List[List[int]]],
                            layer_name: str,
                            neuron_indices: Sequence[int],
                            eps: float = 1e-3,
                            rng: PRNGKey = None
                            ) -> Tuple[Dict[str, Dict[str, jnp.ndarray]], Dict[str, List[List[int]]]]:
    """
    Perform split on `layer_name`'s outputs at indices neuron_indices (referring to *coarse* indices).
    Behavior:
      - Duplicate kernel columns & bias entries for each selected coarse-output group.
      - Append children columns at the end of the kernel (so new fine indices are old_out_dim ..).
      - Perturb duplicated columns by +/- eps*random_dir to break symmetry.
      - Update `mapping` so that the selected coarse index's list-of-fine-indices gains the two new indices.
      - Adjust next-layer kernel input rows: duplicate rows for children and halve them (outgoing split).
    Returns updated (params_new, mapping_new).
    """
    params_u = {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in params.items()}
    # convert mapping to mutable copy
    mapping_u = {k: [list(lst) for lst in v] for k, v in mapping.items()}

    if rng is None:
        rng = jax.random.PRNGKey(0)

    # current layer data
    assert layer_name in params_u
    K = params_u[layer_name]['kernel']  # np array (in_dim, out_dim)
    b = params_u[layer_name]['bias']
    in_dim, out_dim = K.shape
    idx = np.array(neuron_indices, dtype=int)
    m = idx.size
    if m == 0:
        return params_u, mapping_u

    # random directions using numpy for simplicity here
    # (we don't JIT the split operation)
    u = np.random.normal(size=(in_dim, m)).astype(np.float32)
    beta = np.random.normal(size=(m,)).astype(np.float32)

    K_cols = K[:, idx]            # (in_dim, m)
    Ka = K_cols + eps * u
    Kb = K_cols - eps * u
    K_new = np.concatenate([K, Ka, Kb], axis=1)   # new out_dim = out_dim + 2m

    b_cols = b[idx]
    ba = b_cols + eps * beta
    bb = b_cols - eps * beta
    b_new = np.concatenate([b, ba, bb], axis=0)

    params_u[layer_name]['kernel'] = K_new
    params_u[layer_name]['bias'] = b_new

    # update mapping: appended indices range
    appended_start = out_dim
    new_indices = list(range(appended_start, appended_start + 2*m))
    # for each selected coarse index (coarse index = position in mapping list)
    for sel_i_pos, coarse_idx in enumerate(idx.tolist()):
        # mapping element for that coarse index exists; append the two new children fine indices
        mapping_u[layer_name][coarse_idx].extend([new_indices[2*sel_i_pos], new_indices[2*sel_i_pos+1]])

    # adjust next layer's kernel rows if next layer exists
    # next layer name is dense_{i+1}
    assert layer_name.startswith("dense_")
    cur_idx = int(layer_name.split("_")[1])
    next_layer_name = f"dense_{cur_idx+1}"
    if next_layer_name in params_u:
        K_next = params_u[next_layer_name]['kernel']  # (in_next, out_next)
        # rows correspond to outputs of current layer -> duplicate rows for new children, halve them
        rows = K_next[idx, :]    # (m, out_next)
        Ka_rows = 0.5 * rows
        Kb_rows = 0.5 * rows
        K_next_new = np.concatenate([K_next, Ka_rows, Kb_rows], axis=0)
        params_u[next_layer_name]['kernel'] = K_next_new
        # bias unchanged

    # Also for the mapping of the next layer, we need mapping of inputs (not stored). We only store output mapping per layer.
    # The mapping for next layer's coarse outputs remains correct (it references fine indices of the next layer's outputs).
    return params_u, mapping_u

# -------------------------
# Restriction: fine -> coarse projection (averaging groups by mapping)
# -------------------------
def restrict_params_from_fine(params_fine: Dict[str, Dict[str, jnp.ndarray]],
                              mapping_coarse: Dict[str, List[List[int]]]
                              ) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Project a fine-level param dict to a coarse-level param dict using mapping_coarse.
    mapping_coarse[layer_name][j] is list of fine-output indices that correspond to coarse output j.
    Coarse kernel columns are averaged across mapped fine columns.
    For the next-layer kernel, we average rows grouped by mapping_coarse to obtain the smaller input dimension.
    """
    params_coarse = {}
    layer_names = list(mapping_coarse.keys())
    n_layers = len(layer_names)
    for i, layer_name in enumerate(layer_names):
        fine_layer = params_fine[layer_name]
        Kf = np.array(fine_layer['kernel'])
        bf = np.array(fine_layer['bias'])
        groups = mapping_coarse[layer_name]
        # out_dim_coarse = len(groups); in_dim_coarse = Kf.shape[0] (same for both)
        # coarse kernel: for each group, average the Kf[:, group] columns
        Kc_cols = []
        bc = []
        for g in groups:
            cols = Kf[:, g]   # shape (in_dim, len(g))
            avg_col = np.mean(cols, axis=1)
            Kc_cols.append(avg_col)
            bc.append(np.mean(bf[g]))
        Kc = np.stack(Kc_cols, axis=1)   # (in_dim, out_coarse)
        bc = np.array(bc)
        params_coarse[layer_name] = {'kernel': Kc, 'bias': bc}

    # now we must also "compress" the next-layer kernels' input dimension accordingly.
    # For each next-layer, compute K_next_coarse by grouping rows of fine next-layer kernel
    for i in range(n_layers - 1):
        next_name = f"dense_{i+1}"
        if next_name in params_coarse:
            K_next_f = np.array(params_fine[next_name]['kernel'])  # shape (in_fine, out_next)
            # mapping_coarse for layer i gives groups of fine outputs that become coarse outputs; groups length sum = in_fine
            groups = mapping_coarse[f'dense_{i}']
            # for each group, average rows in K_next_f
            rows_grouped = []
            for g in groups:
                rows = K_next_f[g, :]   # (len(g), out_next)
                avg_row = np.mean(rows, axis=0)
                rows_grouped.append(avg_row)
            K_next_c = np.stack(rows_grouped, axis=0)  # (in_coarse, out_next)
            params_coarse[next_name]['kernel'] = K_next_c
            # bias of next layer remains unchanged (coarse next bias already computed above)
    return params_coarse

# -------------------------
# Prolongation: coarse -> fine injection (copy columns into mapped fine indices)
# -------------------------
def prolongate_params_from_coarse(params_coarse: Dict[str, Dict[str, jnp.ndarray]],
                                  mapping_coarse: Dict[str, List[List[int]]],
                                  fine_template_params: Dict[str, Dict[str, jnp.ndarray]]
                                  ) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Build fine-level params by copying coarse columns into fine positions based on mapping_coarse.
    fine_template_params provides the fine shapes (so we know where to write).
    For each coarse output j, fill all fine indices in mapping_coarse[layer][j] with the coarse column.
    For the next-layer rows: for each coarse input row j, copy that row into all corresponding fine rows.
    """
    params_fine = {}
    layer_names = list(mapping_coarse.keys())
    n_layers = len(layer_names)
    for i, layer_name in enumerate(layer_names):
        fine_layer = fine_template_params[layer_name]
        Kf_shape = fine_layer['kernel'].shape
        bf_shape = fine_layer['bias'].shape
        Kf_new = np.zeros_like(fine_layer['kernel'])
        bf_new = np.zeros_like(fine_layer['bias'])
        groups = mapping_coarse[layer_name]
        # coarse column = params_coarse[layer]['kernel'][:, j]
        Kc = np.array(params_coarse[layer_name]['kernel'])
        bc = np.array(params_coarse[layer_name]['bias'])
        for j, g in enumerate(groups):
            for fine_idx in g:
                Kf_new[:, fine_idx] = Kc[:, j]
                bf_new[fine_idx] = bc[j]
        params_fine[layer_name] = {'kernel': Kf_new, 'bias': bf_new}
    # next-layer rows: for i in range(n_layers-1) copy coarse next kernels rows into fine rows
    for i in range(n_layers - 1):
        next_name = f"dense_{i+1}"
        if next_name in params_fine:
            K_next_template = fine_template_params[next_name]['kernel']
            K_next_new = np.zeros_like(K_next_template)
            K_next_coarse = np.array(params_coarse[next_name]['kernel'])  # shape (in_coarse, out_next)
            groups = mapping_coarse[f'dense_{i}']
            for coarse_row_idx, g in enumerate(groups):
                coarse_row = K_next_coarse[coarse_row_idx, :]
                for fine_row in g:
                    K_next_new[fine_row, :] = coarse_row
            params_fine[next_name]['kernel'] = K_next_new
            # bias for next layer already set when copying coarse->fine outputs above
    return params_fine

# -------------------------
# Indicator and Dörfler marking (same as earlier)
# -------------------------
def mse_loss(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((preds - targets)**2, axis=-1)

def neuron_gradient_indicator(params: Dict[str, Dict[str, jnp.ndarray]],
                              loss_fn,
                              batch: Tuple[jnp.ndarray, jnp.ndarray],
                              layer_name: str) -> np.ndarray:
    xb, yb = batch
    # scalar loss to take gradient
    def scalar_loss(p):
        preds = forward(p, xb)
        return jnp.sum(loss_fn(preds, yb))
    grads = jax.grad(scalar_loss)(params)
    g_layer = grads[layer_name]
    gK = g_layer['kernel']
    gb = g_layer['bias']
    gK2 = jnp.sum(gK**2, axis=0)
    eta = jnp.sqrt(gK2 + gb**2)
    return np.array(eta)

def dorfler_marking(eta: np.ndarray, theta: float = 0.6) -> np.ndarray:
    vals = np.array(eta)
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
# Optax training helpers (pure functional)
# -------------------------
def create_optimizer(learning_rate: float = 1e-3):
    return optax.adam(learning_rate)

def init_opt_state(opt, params):
    return opt.init(params)

def apply_gradients(params, opt_state, grads, opt):
    updates, new_opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

def compute_grads(params, batch, loss_fn):
    return jax.grad(lambda p: jnp.mean(loss_fn(forward(p, batch[0]), batch[1])))(params)

# -------------------------
# V-cycle (recursive)
# -------------------------
def v_cycle(levels: List[Dict[str, Any]],
            level_idx: int,
            train_batches: Iterable[Tuple[jnp.ndarray, jnp.ndarray]],
            Sc: int,
            Sf: int,
            loss_fn,
            learning_rate: float = 1e-3):
    """
    Perform V-cycle with recursion on levels[0..L-1] where 0 is coarsest.
    - levels[k] is {'params': params_k, 'mapping': mapping_k, 'template': template_params_k}
      'template' is the fine-shape template used when prolongating from coarse to that fine.
      For level 0 (coarsest) template may equal params itself.
    - level_idx: current level index (0..L-1), typical call with level_idx = L-1 (finest).
    Returns updated params for this level.
    """
    # local convenience
    level = levels[level_idx]
    params = level['params']
    opt = create_optimizer(learning_rate)
    opt_state = init_opt_state(opt, params)

    train_iter = iter(train_batches)

    # pre-smooth on this level
    for _ in range(Sc):
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        grads = compute_grads(params, batch, loss_fn)
        params, opt_state = apply_gradients(params, opt_state, grads, opt)

    # if not coarsest, restrict and recurse
    if level_idx > 0:
        coarse_level = levels[level_idx - 1]
        mapping_coarse = coarse_level['mapping']
        # Restrict current fine params -> coarse params
        params_coarse_init = restrict_params_from_fine(params, mapping_coarse)
        # set coarse params in levels structure for recursion
        levels[level_idx - 1]['params'] = params_coarse_init
        # recursive V-cycle on coarse
        params_coarse_updated = v_cycle(levels, level_idx - 1, train_batches, Sc, Sf, loss_fn, learning_rate)
        # prolongate coarse updated -> fine shapes using template (current level template stores fine's template)
        fine_template = level['template']  # template params for current fine shapes
        params_prolongated = prolongate_params_from_coarse(params_coarse_updated, mapping_coarse, fine_template)
        # inject prolongated params into current params (you may add them as a correction instead, here we REPLACE)
        params = params_prolongated

    # post-smooth on this level
    opt_state = init_opt_state(opt, params)  # re-init optimizer for possibly replaced params
    for _ in range(Sf):
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        grads = compute_grads(params, batch, loss_fn)
        params, opt_state = apply_gradients(params, opt_state, grads, opt)

    # write back
    levels[level_idx]['params'] = params
    return params

# -------------------------
# Top-level AFEM-like outer loop: run V-cycle then mark+refine finest -> append new finer level
# -------------------------
def hierarchical_training_with_vcycle(
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
    """
    outer_iters: number of refine cycles (each cycle runs a V-cycle and then refines the finest level)
    """

    # initialize coarsest level
    rng, sub = jax.random.split(rng)
    params0 = init_params(sub, initial_dims)
    mapping0 = init_mapping_for_dims(initial_dims)
    template0 = {k: {kk: np.array(vv) for kk, vv in layer.items()} for k, layer in params0.items()}
    levels = [{'params': params0, 'mapping': mapping0, 'template': template0}]

    train_iter = iter(train_batches)

    for outer in range(outer_iters):
        print(f"\n=== Outer refinement iter {outer} | levels={len(levels)} ===")
        # run a V-cycle starting at finest level
        finest_idx = len(levels) - 1
        v_cycle(levels, finest_idx, train_batches, Sc, Sf, loss_fn, learning_rate)

        # ESTIMATE on finest level using gradient indicators
        finest = levels[-1]
        params_finest = finest['params']
        eta = neuron_gradient_indicator(params_finest, loss_fn, val_batch_for_estimate, layer_name='dense_0')
        marked = dorfler_marking(eta, theta=theta)
        print("Marked at finest:", marked)

        if marked.size == 0:
            print("No marks -> stopping outer loop")
            break

        # REFINE: split those coarse indices at finest to create a new finer level
        # create new params by splitting current finest params and mapping
        rng_np = np.random.RandomState(int(jax.random.randint(rng, (), 0, 2**31)))
        # the split function earlier used np.random; to keep simple, call it with rng that we ignore
        params_new, mapping_new = split_neurons_in_params(levels[-1]['params'], levels[-1]['mapping'],
                                                         layer_name='dense_0', neuron_indices=marked, eps=split_eps, rng=rng)
        # Build template for new finer level (its template is the array shapes of params_new)
        template_new = {k: {kk: np.array(vv) for kk, vv in layer.items()} for k, layer in params_new.items()}
        levels.append({'params': params_new, 'mapping': mapping_new, 'template': template_new})
        print(f"Appended new finer level -> now levels={len(levels)}")

    return levels

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import itertools

    # Toy dataset
    def batch_gen(batch_size=32):
        rng_np = np.random.RandomState(0)
        while True:
            xb = rng_np.randn(batch_size, 4).astype(np.float32)
            yb = (np.sum(xb**2, axis=1, keepdims=True)).astype(np.float32)
            yield (jnp.array(xb), jnp.array(yb))

    rng = jax.random.PRNGKey(0)
    initial_dims = [4, 32, 1]   # small coarse net
    train_batches = batch_gen()
    val_batch = next(train_batches)

    levels = hierarchical_training_with_vcycle(
        rng,
        initial_dims=initial_dims,
        outer_iters=2,
        train_batches=train_batches,
        val_batch_for_estimate=val_batch,
        Sc=200,
        Sf=100,
        theta=0.6,
        split_eps=1e-3,
        loss_fn=mse_loss,
        learning_rate=1e-3
    )

    print("\nFinal hierarchy sizes:")
    for i, lvl in enumerate(levels):
        print(f" Level {i}:")
        for k, v in lvl['params'].items():
            print(f"   {k}: kernel {np.array(v['kernel']).shape}, bias {np.array(v['bias']).shape}")