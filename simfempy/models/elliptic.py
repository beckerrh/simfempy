from simfempy.models.elliptic_primal import EllipticPrimal
from simfempy.models.elliptic_mixed import EllipticMixed

# ================================================================= #
def Elliptic(**kwargs):
    fem = kwargs.pop("fem", "cr1")
    kwargs["fem"] = fem
    if fem == 'rt0': return EllipticMixed(**kwargs)
    else: return EllipticPrimal(**kwargs)



#=================================================================#
if __name__ == '__main__':
    print("Pas de test")
