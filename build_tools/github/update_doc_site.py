"""Add the built documentation to the site published on GitHub Pages.

The site has a directory for each version of the documentation: ``dev`` for
the development branch and ``X.Y`` for each minor release, built from the tag
of the last release published in it. ``stable`` is a symbolic link to the
newest minor release with a final release, not only pre-releases, and the root
of the site redirects to it, or to ``dev`` until there is one.
``versions.json`` lists the versions for the version switcher of the theme.

Run it from the root of the repository, in a GitHub Actions job started by a
push to a development branch or by a published release, with the built pages
in ``HTML_DIR`` and the current site in ``SITE_DIR``, which it updates.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

DEV = "dev"
STABLE = "stable"

MINOR = re.compile(r"(\d+)\.(\d+)")
FINAL = re.compile(r"\d+\.\d+\.\d+")

REDIRECT = """\
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url={0}/">
    <link rel="canonical" href="{0}/">
  </head>
</html>
"""


def minor_key(name):
    """Return the sort key of the directory of a minor release."""
    return tuple(int(part) for part in name.split("."))


def main():
    html_dir = Path(os.environ["HTML_DIR"])
    site_dir = Path(os.environ["SITE_DIR"])
    site_url = os.environ["SITE_URL"]
    ref_name = os.environ["GITHUB_REF_NAME"]

    if os.environ["GITHUB_EVENT_NAME"] == "release":
        if not (match := MINOR.match(ref_name)):
            sys.exit(f"The tag of the release is not a version: {ref_name!r}.")
        version = ".".join(match.groups())
        final = FINAL.fullmatch(ref_name) is not None
    else:
        version, final = DEV, False

    target = site_dir / version
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(html_dir, target)

    stable = site_dir / STABLE
    # A release of an older minor release does not take the link away from a
    # newer one, so it only moves forward
    if final and (
        not stable.is_symlink() or minor_key(version) >= minor_key(os.readlink(stable))
    ):
        stable.unlink(missing_ok=True)
        stable.symlink_to(version, target_is_directory=True)

    stable_version = os.readlink(stable) if stable.is_symlink() else None
    minors = sorted(
        (path.name for path in site_dir.iterdir() if MINOR.fullmatch(path.name)),
        key=minor_key,
        reverse=True,
    )
    versions = [{"name": DEV, "version": DEV, "url": f"{site_url}/{DEV}/"}]
    for minor in minors:
        if minor == stable_version:
            versions.append(
                {
                    "name": f"{minor} ({STABLE})",
                    "version": minor,
                    "url": f"{site_url}/{STABLE}/",
                    "preferred": True,
                }
            )
        else:
            versions.append(
                {"name": minor, "version": minor, "url": f"{site_url}/{minor}/"}
            )
    (site_dir / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")

    root = STABLE if stable_version else DEV
    (site_dir / "index.html").write_text(REDIRECT.format(f"{site_url}/{root}"))
    # Serve the files as they are, without building the site with Jekyll,
    # which leaves out the directories that start with an underscore, such
    # as the static files of Sphinx
    (site_dir / ".nojekyll").touch()


if __name__ == "__main__":
    main()
