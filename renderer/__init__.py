from renderer.gsplat_renderer import GSplatRenderer



def create_renderer(renderer_type, **kwargs):
    if renderer_type == "gsplat":
        return GSplatRenderer(**kwargs)
    else:
        return GSplatRenderer(**kwargs)
