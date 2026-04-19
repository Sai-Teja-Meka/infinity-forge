"""Frozen canonical probe values for behavioral fingerprinting (Layer 6).

These 20 values per input type are the inputs every candidate atom is
evaluated against to produce its behavioral fingerprint. They are FROZEN
FOREVER. Changing any value here invalidates every behavioral fingerprint
already stored in the library, because two atoms can only be compared if
they were run against the same probes.

The values were generated once by running `inputs.sample_input(t, seed)`
for `seed in range(20)` across all 8 input types, via the helper script
`scripts/freeze_probes.py`. The resulting literals were pasted here. The
runtime never re-samples. If `inputs.py` changes in the future, these
literals remain unchanged.

Seeds 0..4 for scalar types (int/float/str/bool) yield canonical edge
cases pinned in `inputs.py`; seeds 5..19 come from the deterministic RNG
path. See `test_probes.py` for the pinned edge-case assertions.

`EXPECTED_OUTPUT_TYPES` is the shallow Python type each output-type string
is expected to produce. The check is shallow by design: `list[int]` maps
to `list` — we do not recurse into element types on Day 4. An atom that
returns a `list` of mixed types still passes the output-type gate; the
fingerprint will distinguish it from a same-signature atom behaviorally.
"""
from __future__ import annotations

PROBES: dict[str, list] = {
    'int': [0, 1, -1, 2, 10, 59, 46, -18, -42, 18, 46, 15, 21, -34, -73, -47, -8, 33, -54, 73],
    'float': [0.0, 1.0, -1.0, 0.5, -0.5, 24.5803, 58.668, -35.2334, -54.6588, -7.3985, 14.2805, -9.5241, -5.0859, -48.1983, -78.6343, 93.0484, -27.6954, 4.3968, -63.747, 35.4252],
    'bool': [True, False, True, False, True, False, False, True, True, True, False, True, True, True, True, False, True, False, True, False],
    'str': ['', 'a', 'hello', 'AbC', 'hi world', 'qVwYS81VP7Hb1DX8pPd', '0fFWqcajQLE9WVxuXb', '8jzPde0Igx', 'x9yimTc', 'Nxril3RavGD5Mf', 'cBEKanD0F0rPZkcHFu', '3J27XDCG2LmlZG', 'rQHQwjyaxErPZDS', 'sR6RZ24l', 'NSW', 'aHVck6', 'EEsAoCaA2QT', 'AZt9xslXTTIQrh6b', 'hQCvp', 'Y'],
    'list[int]': [[94, 7, -90, -34, 30, 24, 3, 100, -23, 22, -9, 49], [45, 95, -84, -35], [-77], [51, 39, -67, -6, 54, 21, 60], [-23, -74, 84, 1, 22, -61, -77], [-35, 89, -9, 76, 89, 66, 35, -93, 19, 98, -37, 66, -87, -60, -72, -5, 20, -37, -3], [-80, 24, 95, -34, -91, -100, -63, 69, 50, 20, 94, 88, -5, -19, 97, -95, -31, 25], [-62, 1, 66, -88, -82, 37, -76, -7, 49, -86], [-6, -4, -68, -51, 80, -89, -79], [56, -5, -32, -65, -53, 73, -99, -14, 28, 18, 54, -80, -15, 41], [-92, 9, 23, 47, -97, -48, 18, 25, -29, 67, -59, -92, 33, 25, -17, -81, -37, 90], [43, 99, 19, 15, 30, 50, -52, -53, 31, 21, 61, 57, -53, -76], [-32, 68, 35, 70, -11, -64, -3, -98, -5, 23, -30, 64, 17, 76, 53], [-26, 75, 75, -53, 66, -41, 70, -63], [57, 79, 93], [-98, 33, 88, -91, -60, -39], [20, 23, -28, 6, -42, 14, -99, 4, 68, 82, -34], [6, -23, -7, -26, -56, 96, 80, 80, 38, 69, -29, -72, -94, -37, -2, 91], [-69, 69, 14, -15, -39], [100]],
    'list[float]': [[51.5909, -15.8857, -48.2166, 2.2549, -19.0132, 56.7597, -39.3375, -4.6806, 16.6764, 81.6226, 0.9374, -43.6324], [13.8408, 60.453, -87.3786, -76.4163], [-81.683], [18.5282, -73.9154, 83.189, -5.1893, 16.1704, 21.1199, 81.7637], [-39.3403, 44.2434, -4.2243, -81.98, -96.0366, 9.8804, -42.1248], [-48.911, -28.2929, 38.0894, 68.3022, 30.4063, 6.008, 68.0696, 55.1917, -50.1895, -89.6293, -68.6297, -25.6413, 73.6891, -23.8484, -79.6051, -50.1339, 46.2367, -18.3698, -63.5849], [64.3908, -2.9931, -47.6757, -99.9097, 32.5637, -5.9491, 51.9461, -25.3679, 54.028, -45.4604, 60.3831, 45.965, -17.1987, 7.661, 36.4103, -61.403, 10.723, 61.0248], [89.5731, -21.0353, -90.3427, 64.2549, -81.174, 16.5576, 81.9408, -57.0604, -82.8106, -16.3656], [-25.9176, -24.9261, -61.3767, -91.2443, -72.6314, 62.2528, 1.2721], [22.643, -46.5731, -62.771, 35.3287, -32.3349, -7.2557, 20.9398, -33.1952, 87.2772, 40.0197, 45.6384, -66.1172, 89.2431, 90.2542], [-93.483, -3.4877, -97.0335, -7.4839, -1.7407, -44.4959, 62.1238, -93.1193, -1.9829, -84.7821, 90.4798, 49.1093, -27.7683, -15.9208, -72.2621, -28.9837, -15.7472, 65.3696], [73.1485, 71.3726, 56.2436, -9.6335, 71.0091, -62.0196, 60.7801, -4.8474, 22.7918, -62.7653, -10.6814, -71.641, 7.7387, 78.0761], [-46.201, 5.8262, -30.0429, -23.6718, 99.3244, -3.4922, 28.6818, -7.9563, 72.8689, -54.4711, -99.6668, 24.8466, 95.6081, -26.4689, -32.064], [-41.8485, 81.908, 60.4009, 77.9996, 30.4019, 33.2371, 73.8462, 28.1914], [23.1757, 51.0411, 5.4224], [-97.6691, 47.1983, -68.3975, 97.2679, -96.6239, 75.8983], [-6.1555, -43.0133, -54.6711, -98.8293, 70.8273, 42.2003, -52.4283, -55.514, -40.6805, 62.9822, 33.343], [-17.1664, -39.3153, -26.8753, -65.0623, 41.0285, 8.0371, -44.3447, 83.4986, -50.2194, 63.0372, -16.0867, 71.941, 60.4748, 28.105, 36.8869, 44.8545], [-75.4369, -10.2349, -52.1007, 90.3826, 25.6376], [56.9823]],
    'list[str]': [['cq9GFz6Y1t9Ew', '56nGisiWgNZq6I TZM', 'tgUe', 'EJgwBuNO6n', 'EC3HqdZ6J6afU1zT0', 'aNF03vpUuT3em6KopZ9j', 'Cffu4G7FgtJsThJv0', 'n9ZMJLsCfMZyuKpsl', '0lcN Q', 'EefRWi4j', '1', '5S'], ['2ZWeqhFWCEPyYngFb5', 'BMWXaSCrUZoL', '5ub', ''], ['fx'], ['Iix6MEOLeMa61EqJom', 'I1JEzO3joOj37Hy', '', 'kW', 'ctXb03rEMU64yTY6Bz', 'C97i4xgciFnq9RBXO2', 'AG1yKwILA'], ['gUzEjfebz', '6sZWdoHIxrXl0gqn8', '', 'ZqZrmktsO3U9224xf2Mv', 'GplpErf87038', '1ta6sKT4t2WGmABMs', 'Ckotq0ZcfcDOr'], ['VwYS81VP', 'b1DX8pPd5khxE3py', 'gKpaUnArl63XykWZe', 'NNCi', 'a 3a', 'Xn9 k3', 'su9mI', 'nl89Sm995ytbxAk7jqev', '0MLaMRTve', 'w0tESulEE', 'dq 8b', '2zbJYAxyL1a', 'cTlN99mhWp708D', 'Gw5HqXDgLVX', '3scB8 fnvGN', '6jvr7SIftRu', 'lZfOjUStE', 'UdfMI', 'cpVMw0qDPAjd'], ['FW', 'cajQLE9W', 'uXbrFZmU3A6', 'IRgmKJSZUqQZNRf2B', 'fxZAZqCSgW', 'SOYsg8', 'L', '0P6xF7', 'GKP S5', ' 6bOxpMBtwLhfG4R', 'mhMQrtUmyEoiMn13', 'amXkbPvJ5QNNtxyH', 'siRFdlBMVzg ', 'pZ f5MQ34CCY1y', 'HB', 'tT0A8fmVRrCFUY9', 'bbIhq', 'ZxmqGCv7Hq A74ANFr'], ['zPde', 'gxLd6GncfBAepfJBd', 'h8oOOL8dKLzd ocJ2i', 'AjIhKtJ0R', 'gLKOm', 'gJTeKdNnFRI', 'XuDL7DxtpYlSX', 'fKtHF4v', 'sM ehGAkWvj7FA', '9'], ['9yimTcfipZG', 'zPbDFD', 'FKm51zfFoWbS', 'HAE56yUh', 'g0eyN1yg', 'v', 'SfF5PH5'], ['xril3RavGD5MfvJ7NSc', 'kT8C8UBkkpdh', 'G37L', 'XS', 'YV4g6snRoUYA', 'Xr', 'zrvZcm', '', 'd5y2FibpBV62h', 'ahWLm52mva5fiI6bGfK', 'I6mAezmOWfSLjl8', 'U9cdrJRM2jVrVKch3Tz', 'kNGcVx2', 'LKXSf3wh34LxCnzm'], ['B', 'KanD0F0rPZkcHFu', 'p8', 'cA3iMwyAs0R', 'DlRtQxiD', ' CNycLa', 'im86tIx', 'uQJCBEe', 'Lu2Gk1o ApccFt0MQeI7', 'jy', '8x6Mjh9XXgCk Zm8wB', 'CpRrjNHl3hrDt', 'QP80l', 'EXwuB', 'aITcv5u', 'fqCzLky', 'R5pVH6rHEMFekFR', '5ziAILwIyFSkJC'], ['27XDCG2LmlZGEONYl', 'Ctj', 'IZ', 'cM z9CPVNPkNa1Hedcm4', 'MbXDuCL', 'HoOsFa', 'DP', 'AJ71fTqu', 'GsbeKXg', 'g2sye9b2Rann', 'E', ' TzAeKOmXRrv', 'tv', ''], ['QHQwjyax', 'rPZDS3MoJaQNj C', 'kv5ndK0meGR', 'RzZ1fb6d06', 'ofB8ChQB iIu4NJk', 'J', 'G0fzM', 'QMEEMyIbPUf9m', 'w8x3SyRt', 'qpv', 'xGKZWGlby02', 'cHboRBcynXMgW', 'oleSrcBrFwMOUdGDx', 'vsDES9', 'p9kD7JxlmXVo6Ma'], ['R6RZ24lPo', '3oPU', 'ieI2n', 'bBi1RMar1', 'f3YZ', '0CVB8iY4', 'w2oF5WJK', 'Qx4BOuPhw0MZO'], ['SWPH8prVqsUeQCtDR3z', 'X6hqo35uwZqx', 'OHjkJQQrkaPehMvbfrn2'], ['', 'Vck6pbd4ZRj2Sxph', 'DTwrzqwo72', '3wZuot', 'AoKD1AFfDKZxCKu7', 'CzX3ZeF981bmiIlN8YS2'], ['EsAoCaA2QTqpOoa', 't0vQj8VMt', '', '9Mqb5jZ', 'QObDDMOTsoYtxqAYfwF', 'HPl8KsLcs f1Y', '', 'xpFjttuDDekSEU2a', '13Fa61ESYhD1Nf', 'Pb9jTo6z5xcIcQP', 'MuEGQ80YRP10'], ['Zt9xslXTTIQrh', '', 'y0VAq3G', 'O2R8UziJdi', 'j4TIJZ', 'vIh4TO', 'tA', 'G8', 'OMjRZA0G6vbBxKd', 'd9ExLXa3zph', 'n9pH9xdreYrmVM1JI', '5iqQt6wukvg6KLYrv', '', 'W', 'bDVrEOdUmtq', 'VT'], ['QCv', 'm8FOFlE', 'D4qmq5Shu', 'RWYl398ZpkpmVxKG', 'ZR53W1'], ['hGmzwHsLjMqgqAu9']],
    'dict': [{'g': 'z6Y1t9EwL56nGis', 'a': -76, 'c': '6I TZM5j'}, {'b': 26}, {}, {'c': 54}, {'e': 1}, {'e': 19, 'f': 'Pd5khxE', 'c': 'IgKpaUnArl63', 'h': 'WZeiN'}, {'b': 69, 'd': 'uXbrFZmU3A6', 'c': 'mKJ', 'a': 'QZNRf2Bv'}, {'c': '0I', 'd': 49}, {'f': 'mTcf'}, {'f': 73, 'c': 28, 'b': 54}, {'a': 18, 'd': 'PZkcHFue', 'g': 90, 'e': 'A'}, {'h': 'lZGEON', 'd': 'Ctj', 'e': 77}, {'e': 'yaxE', 'f': 17, 'h': 'oJaQNj Cxkv5ndK0meG'}, {'e': 'PoQj3', 'f': 87}, {}, {'a': 'k'}, {'h': -42, 'd': 4}, {'g': 96, 'h': 'Qrh6bpy0VAq3GZuO2', 'c': 'iJdi0Y4mj4TI', 'f': 'vIh4TO'}, {'b': 'pm8FOFlEsD'}, {'a': 35, 'g': -63, 'e': 'qAu', 'h': 'Xu5', 'b': 45}],
}


EXPECTED_OUTPUT_TYPES: dict[str, type] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "list[int]": list,
    "list[float]": list,
    "list[str]": list,
    "dict": dict,
}
