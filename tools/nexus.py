class Settings:
    timeout = 600
    ta_threshold = 5
    verifiers = ["DNNV:eran_deeppoly", "DNNV:nnenum", "DNNVWB:neurify", "DNNV:marabou"]
    fig_dir = "figs"

    convert_names = {
        "Baseline": "Baseline",
        "BS_SDD": "B_SDD",
        "BS_SAD": "B_SAD",
        "BS_NIP": "B_NIP",
        "BS_SIP": "B_SIP",
        "RS_SDD": "R_SDD",
        "RS_SAD": "R_SAD",
        "RS_NIP": "R_NIP",
        "RS_SIP": "R_SIP",
        "PR_SDD": "P_SDD",
        "PR_SAD": "P_SAD",
        "PR_NIP": "P_NIP",
        "PR_SIP": "P_SIP",
        "BRS_SDD": "BR_SDD",
        "BRS_SAD": "BR_SAD",
        "BRS_NIP": "BR_NIP",
        "BRS_SIP": "BR_SIP",
        "BSPR_SDD": "BP_SDD",
        "BSPR_SAD": "BP_SAD",
        "BSPR_NIP": "BP_NIP",
        "BSPR_SIP": "BP_SIP",
        "RSPR_SDD": "RP_SDD",
        "RSPR_SAD": "RP_SAD",
        "RSPR_NIP": "RP_NIP",
        "RSPR_SIP": "RP_SIP",
        "BRSPR_SDD": "BRP_SDD",
        "BRSPR_SAD": "BRP_SAD",
        "BRSPR_NIP": "BRP_NIP",
        "BRSPR_SIP": "BRP_SIP",
    }

    heuristics = {
        "e1": [
            "Baseline",
            "B_SDD",
            "B_SAD",
            "B_NIP",
            "B_SIP",
            "R_SDD",
            "R_SAD",
            "R_NIP",
            "R_SIP",
            "P_SDD",
            "P_SAD",
            "P_NIP",
            "P_SIP",
        ],
        "e2": [
            "Baseline",
            "BR_SDD",
            "BR_SAD",
            "BR_NIP",
            "BR_SIP",
            "BP_SDD",
            "BP_SAD",
            "BP_NIP",
            "BP_SIP",
            "RP_SDD",
            "RP_SAD",
            "RP_NIP",
            "RP_SIP",
            "BRP_SDD",
            "BRP_SAD",
            "BRP_NIP",
            "BRP_SIP",
        ],
        "e3": [
            "Baseline",
            "BR_SDD",
            "BR_SAD",
            "BR_NIP",
            "BR_SIP",
            "BP_SDD",
            "BP_SAD",
            "BP_NIP",
            "BP_SIP",
            "RP_SDD",
            "RP_SAD",
            "RP_NIP",
            "RP_SIP",
            "BRP_SDD",
            "BRP_SAD",
            "BRP_NIP",
            "BRP_SIP",
        ],
    }

    """
    excludes = {
        "e1": {"B_NIP": ["CIFAR10"], "B_SIP": ["CIFAR10"], "R_SIP": ["CIFAR10"]},
        "e1_": {"B_NIP": ["CIFAR10"], "B_SIP": ["CIFAR10"], "R_SIP": ["CIFAR10"]},
        "e2": {
            #
            "BR_SDD": [],
            "BR_SAD": [],
            "BR_NIP": ["MNIST", "FashionMNIST", "CIFAR10"],
            "BR_SIP": ["FashionMNIST", "CIFAR10"],
            #
            "BP_SDD": ["MNIST", "CIFAR10"],
            "BP_SAD": ["MNIST"],
            "BP_NIP": ["MNIST", "FashionMNIST", "CIFAR10"],
            "BP_SIP": ["MNIST", "FashionMNIST", "CIFAR10"],
            #
            "RP_SDD": ["CIFAR10"],
            "RP_SAD": [],
            "RP_NIP": ["CIFAR10"],
            "RP_SIP": ["CIFAR10"],
            #
            "BRP_SDD": ["MNIST", "FashionMNIST", "CIFAR10"],
            "BRP_SAD": [
                "MNIST",
            ],
            "BRP_NIP": ["MNIST", "FashionMNIST", "CIFAR10"],
            "BRP_SIP": ["MNIST", "FashionMNIST", "CIFAR10"],
        },
        "e2_": {
            #
            "BR_SDD": [],
            "BR_SAD": [],
            "BR_NIP": ["MNIST", "FashionMNIST", "CIFAR10"],
            "BR_SIP": ["FashionMNIST", "CIFAR10"],
            #
            "BP_SDD": [
                "MNIST",
            ],
            "BP_SAD": ["MNIST"],
            "BP_NIP": ["MNIST", "FashionMNIST", "CIFAR10"],
            "BP_SIP": ["MNIST", "FashionMNIST", "CIFAR10"],
            #
            "RP_SDD": ["CIFAR10"],
            "RP_SAD": [],
            "RP_NIP": ["CIFAR10"],
            "RP_SIP": ["CIFAR10"],
            #
            "BRP_SDD": ["MNIST", "FashionMNIST", "CIFAR10"],
            "BRP_SAD": [
                "MNIST",
            ],
            "BRP_NIP": ["MNIST", "FashionMNIST", "CIFAR10"],
            "BRP_SIP": ["MNIST", "FashionMNIST", "CIFAR10"],
        },
        "e3": {
            "BR_NIP": ["CIFAR10"],
            "BR_SIP": ["CIFAR10"],
            "BP_NIP": ["CIFAR10"],
            "BP_SIP": ["CIFAR10"],
            "RP_SDD": ["CIFAR10"],
            "RP_NIP": ["CIFAR10"],
            "RP_SIP": ["CIFAR10"],
            "BRP_SDD": ["CIFAR10"],
            "BRP_NIP": ["MNIST", "FashionMNIST", "CIFAR10"],
            "BRP_SIP": ["CIFAR10"],
        },
    }
    """
