def replace_submodule(model, module_name, new_module):
    """
    module_name:
        "layers.0.attn"

    new_module:
        nn.Module
    """

    parts = module_name.split(".")

    parent = model

    # 找到父module
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)

    last = parts[-1]

    # 替换
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)