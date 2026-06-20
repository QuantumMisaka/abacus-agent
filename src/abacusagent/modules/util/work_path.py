import os
import traceback


def _slugify_work_note(value) -> str:
    text = "" if value is None else str(value).strip()
    parts = []
    previous_dash = False
    for char in text:
        if char.isalnum() or char in "_-":
            parts.append(char)
            previous_dash = False
        else:
            if not previous_dash:
                parts.append("-")
                previous_dash = True
    return "".join(parts).strip("-")


def _random_4digit_suffix() -> str:
    return f"{int.from_bytes(os.urandom(2), 'big') % 10000:04d}"


def generate_work_path(note=None, fallback_name=None, create: bool = True) -> str:
    """
    Generate a note-aware working directory name.

    Naming rule: ``<slug>-<4digit>``. The slug comes from ``note`` when
    provided. Legacy internal callers without ``note`` fall back to
    ``fallback_name`` or the calling function name.

    Args:
        note: Human-readable task label used for the directory slug. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.
        fallback_name: Compatibility slug source for internal callers without note.
        create: Whether to create the returned directory.

    Returns:
        str: Relative working directory path.
    """
    if note is None:
        base = fallback_name or traceback.extract_stack(limit=2)[-2].name
    else:
        base = note

    slug = _slugify_work_note(base)
    if not slug:
        raise ValueError("note slug is empty; provide a note containing letters, numbers, '_' or '-'.")
    if len(slug) > 64:
        slug = slug[:64].rstrip("-")

    for _ in range(100):
        work_path = f"{slug}-{_random_4digit_suffix()}"
        if os.path.exists(work_path):
            continue
        if create:
            try:
                os.mkdir(work_path)
            except FileExistsError:
                continue
        return work_path

    raise FileExistsError(f"Unable to create a unique work directory for note '{slug}' after 100 attempts.")
