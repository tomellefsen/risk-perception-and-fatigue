from models.registry import MODEL_REGISTRY

def standardize_result(model_name, t, compartments, meta):
    """
    Convert (t, compartments, meta) into the standard result dict format.
    Uses meta['compartments'] if provided, otherwise falls back to
    MODEL_REGISTRY[model_name]['compartments'], and finally generic Var names.
    """
    #Check if the model itself provided compartment names
    names = meta.get("compartments")

    #Otherwise, fall back to registry definition
    if names is None:
        registry_info = MODEL_REGISTRY.get(model_name, {})
        names = registry_info.get("compartments")

    #Otherwise, generic names
    if names is None:
        names = [f"Var{i}" for i in range(len(compartments))]

    #Ensure names match compartments length
    names = names[:len(compartments)]

    series = {name: values for name, values in zip(names, compartments)}

    return {
        "time": t,
        "series": series,
        "meta": meta
    }
