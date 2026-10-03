"""Standalone sitecustomize copied into plotting subprocesses.

Generated scripts may reset rcParams or select unavailable Chinese fonts.
Apply the bundled font to CJK Text before layout, so both text measurement
and rendering use the same glyphs. Font size and other Text settings survive.
This file must not import project modules: the sandbox only receives this file.
"""
import os


def _has_cjk(text):
    return any(
        "\u2e80" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        or "\uff00" <= char <= "\uffef"
        or "\U00020000" <= char <= "\U000323af"
        for char in text
    )


def install_font_fallback():
    font = os.environ.get("AUTOREPRO_FONT_PATH", "")
    if not os.path.isfile(font):
        return
    try:
        from matplotlib import font_manager, rcParams
        from matplotlib.text import Text
    except ImportError:
        return

    font_manager.fontManager.addfont(font)
    name = font_manager.FontProperties(fname=font).get_name()
    rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    if getattr(Text._get_layout, "_autorepro_cjk", False):
        return

    original_layout = Text._get_layout

    def layout_with_cjk_font(self, renderer):
        if _has_cjk(self.get_text()) and self.get_fontproperties().get_file() != font:
            properties = self.get_fontproperties().copy()
            # A file path also overrides an explicit incompatible fname.
            properties.set_file(font)
            self.set_fontproperties(properties)
        return original_layout(self, renderer)

    layout_with_cjk_font._autorepro_cjk = True
    Text._get_layout = layout_with_cjk_font


install_font_fallback()
