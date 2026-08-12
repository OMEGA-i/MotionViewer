"""Blender-side rendering helpers.

Most modules in this package are importable outside Blender for tests, but
functions that touch `bpy` import it lazily and fail with setup-oriented errors.
"""
