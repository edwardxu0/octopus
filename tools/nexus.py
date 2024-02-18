from pathlib import Path

class Settings:
    timeout = 300
    fig_dir = "results_figs"
    
    Path(fig_dir).mkdir(exist_ok=True, parents=True)
    

    verifier_convert_names = {
        "SH:abcrown": "α-CROWN",
        "SH:abcrown2": "α-β-CROWN",
        "SH:mnbab": "MN-Bab",
        "SH:nnenum": "NNEnum",
    }

    verifier_order = ["α-CROWN", "α-β-CROWN", "MN-Bab", "NNEnum"]

    network_conver_names = {
        "Net256x2": "M2",
        "Net256x6": "M6",
        "CIFAR2020_8_255": "C3",
    }
    network_order = ["M2", "M6", "C3"]

    stabilizer_convert_names = {
        "Baseline": "Baseline",
        "RS_SDD": "RS_SDD",
        "RS_SAD": "RS_SAD",
        "RS_NIP": "RS_NIP",
        "RS_SIP": "RS_SIP",
        "RS_ALR": "RS_ALR",
        "RS_ALRo": "RS_ALRo",
        "BS_SDD": "BS_SDD",
        "BS_SAD": "BS_SAD",
        "BS_NIP": "BS_NIP",
        "BS_SIP": "BS_SIP",
        "BS_ALR": "BS_ALR",
        "BS_ALRo": "BS_ALRo",
        "SP_SDD": "SP_SDD",
        "SP_SAD": "SP_SAD",
        "SP_NIP": "SP_NIP",
        "SP_SIP": "SP_SIP",
        "SP_ALR": "SP_ALR",
        "SP_ALRo": "SP_ALRo",
    }

    stabilizers_order = [
        # singletons
        "RS_SDD",
        "RS_SAD",
        "RS_NIP",
        "RS_SIP",
        "RS_ALR",
        "RS_ALRo",
        "BS_SDD",
        "BS_SAD",
        "BS_NIP",
        "BS_SIP",
        "BS_ALR",
        "BS_ALRo",
        "SP_SDD",
        "SP_SAD",
        "SP_NIP",
        "SP_SIP",
        "SP_ALR",
        "SP_ALRo",
        "Baseline",
    ]
