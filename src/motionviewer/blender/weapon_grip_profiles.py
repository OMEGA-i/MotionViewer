"""Measured, asset-scoped hand poses; contains no source mesh vertices."""

PROFILES = [
    {
        "id": "feixiao-right-blue-shaft-v1",
        "hand_signature": "e66c65ecc65aa9dcfc3ce3f21aae965617c6d8ff0cc4a0c43e1d36af6283a7f5",
        "weapon_signature": "f195789f7e020f8e93e171d32928b0551baa487dd8e91b37db2af087ba9c1add",
        "source_weapon_signature": "bc44eaa2686036c5339115b50283cf8bdced5d260613e7463a185553df8490ca",
        "component_selection": {
            "local_axis": 0,
            "absolute_limit_m": 0.2,
            "label": "Central two-handed axe; exclude the two independent side weapons",
        },
        "hand_bone": "右手首",
        "grip": "high",
        "grip_center_local": [0.0, 0.62, 0.0],
        "weapon_in_hand": [
            [-0.06186442822217941, -0.993952214717865, 0.09073127806186676, 0.5289137959480286],
            [0.9973496198654175, -0.05807486176490784, 0.04383079707622528, 0.08995699882507324],
            [-0.03829646855592728, 0.09320239722728729, 0.9949102997779846, -0.027668898925185204],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "finger_eulers_degrees": {
            "右人指１": [87.99999237060547, 0.4265204966068268, -1.1400727033615112],
            "右人指２": [76.0, 0.7190975546836853, -1.9717371463775635],
            "右人指３": [61.0, -0.1397327333688736, -1.5031392574310303],
            "右中指１": [90.00000762939453, 0.15466418862342834, -1.2650222778320312],
            "右中指２": [68.0, 0.34731996059417725, -2.2111828327178955],
            "右中指３": [63.999996185302734, -0.2732403576374054, -1.5798537731170654],
            "右薬指１": [85.0, -0.27280011773109436, -1.368391513824463],
            "右薬指２": [73.99998474121094, -0.02021900750696659, -2.3528170585632324],
            "右薬指３": [62.999996185302734, -0.12767621874809265, -1.6920589208602905],
            "右小指１": [78.0, -0.9582259654998779, -1.5303694009780884],
            "右小指２": [72.0, -1.1847777366638184, -2.756875991821289],
            "右小指３": [66.0, 0.046321142464876175, -1.7897077798843384],
            "右親指０": [-6.752132435394742e-07, 39.999996185302734, -90.0],
            "右親指１": [-80.0, 3.3191094398498535, -0.5526760816574097],
            "右親指２": [-90.00000762939453, 2.3333687782287598, -0.4702705144882202],
        },
        "shaft_mm": 29.6,
        "length_m": 2.320050358772278,
        "grip_section": "Measured stable blue shaft (about 29 x 30 mm), away from the 38–47 mm decorative "
        "flare; asset scale unchanged.",
        "thumb_contact": "Opposition toward the index finger with negative local-X thumb flexion; calibrated "
        "against hand and shaft surfaces.",
        "calibration_motion": "active/128701.smplx.npz",
        "calibration_method": "Numerical skin/weapon triangle contact refinement followed by two-angle render "
        "review; asset-specific artistic pose.",
        "left_arm_clearance": {
            "axis_in_weapon_frame": [-0.23919492, -0.93800425, -0.25086334],
            "degrees": 17.0,
            "transition_seconds": 0.4,
            "max_offset_m": 0.12,
            "max_joint_degrees": 20.0,
        },
    }
]
