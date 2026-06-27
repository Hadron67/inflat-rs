def next_unique_name(prefix: str, used_names: set[str]) -> str:
    i = 0
    while True:
        name = prefix + str(i)
        if name not in used_names:
            used_names.add(name)
            return name
        i += 1
