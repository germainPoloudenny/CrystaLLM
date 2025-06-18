def generate_hkl_list(hkl_max_index):
    return [
        (h, k, l)
        for h in range(-hkl_max_index, hkl_max_index + 1)
        for k in range(-hkl_max_index, hkl_max_index + 1)
        for l in range(0, hkl_max_index + 1)
        if not (h == 0 and k == 0 and l == 0)
    ]
