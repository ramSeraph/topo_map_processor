#!/bin/bash

# A script to upload files from a folder to a GitHub release,
# skipping files that already exist in the release.
# If the fourth argument is 'yes', it will overwrite existing assets.

set -e
set -o pipefail

# --- Configuration ---
# TAG: The git tag of the release to upload to.
# FOLDER: The local folder containing the files to upload.
# OVERWRITE: Optional. Set to 'yes' to allow overwriting existing assets.
prog_name=${COMMNAND_NAME:-$(basename "$0")}

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $prog_name <tag> <folder> <extension_without_the_leading_dot> [yes_to_overwrite]"
    exit 1
fi

TAG="$1"
FOLDER="$2"
EXT="$3"
OVERWRITE_FLAG=${4:-no}

# --- Pre-flight Checks ---

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "Error: gh command-line tool is not installed. Please install it to continue."
    exit 1
fi

if [ -z "$TAG" ]; then
    echo "Error: Release tag is not specified."
    exit 1
fi

if [ ! -d "$FOLDER" ]; then
    echo "Error: Folder '$FOLDER' not found."
    exit 1
fi

if [ -z "$EXT" ]; then
    echo "Error: Extension '$EXT' is not specified."
    exit 1
fi

ALLOW_OVERWRITE=0
if [ "$OVERWRITE_FLAG" == "yes" ]; then
    ALLOW_OVERWRITE=1
    echo "Overwrite flag is set. Existing assets will be overwritten."
fi

# --- Main Logic ---

echo "Fetching existing assets for releases matching pattern '${TAG}(-extra[0-9]+)?'..."

# Get all releases, then filter for the base release and any -extra releases, and sort them
RELEASES_TO_PROCESS=$(gh release list --json tagName -q '.[].tagName' | grep -E "^${TAG}(-extra[0-9]+)?$" | sort -V)

if [ -z "$RELEASES_TO_PROCESS" ]; then
    echo "Error: No releases found matching pattern '${TAG}(-extra[0-9]+)?'." >&2
    exit 1
fi

echo "Found releases: $RELEASES_TO_PROCESS"

# Store assets with their release tags. Format: "asset_name release_tag"
EXISTING_ASSETS_WITH_RELEASES=""
for release in $RELEASES_TO_PROCESS; do
  echo "Fetching assets from release: $release"
  assets=$(gh release view "$release" --json assets -q '.assets[].name' 2>/dev/null || echo "")
  if [ -n "$assets" ]; then
    while IFS= read -r asset_name; do
      if [ -n "$asset_name" ]; then
        EXISTING_ASSETS_WITH_RELEASES="${EXISTING_ASSETS_WITH_RELEASES}${asset_name} ${release}"$'
'
      fi
    done <<< "$assets"
  fi
done

if [ -z "$EXISTING_ASSETS_WITH_RELEASES" ]; then
    echo "Could not fetch any assets. This could mean the releases do not have assets, or you lack permissions."
fi

# Determine available releases and their current asset counts
AVAILABLE_RELEASES=()
AVAILABLE_ASSET_COUNTS=()
for release in $RELEASES_TO_PROCESS; do
    count=$(gh release view "$release" --json assets -q '.assets | length' 2>/dev/null || echo "0")
    
    MAX_ASSETS=998
    if [ "$release" == "$TAG" ]; then
        MAX_ASSETS=988
    fi

    if [ "$count" -lt "$MAX_ASSETS" ]; then
        AVAILABLE_RELEASES+=("$release")
        AVAILABLE_ASSET_COUNTS+=("$count")
    fi
done

echo "Starting upload process from folder '$FOLDER'..."

# Find all files in the folder that are not yet in any release and loop through them.
find "${FOLDER}" -type f | grep "^.*\.${EXT}$" | while read -r FILE_PATH; do
    FILENAME=$(basename "$FILE_PATH")

    RELEASE_FOR_ASSET=$(echo -n "$EXISTING_ASSETS_WITH_RELEASES" | grep "^${FILENAME} " | head -n 1 | awk '{print $2}')

    if [ -n "$RELEASE_FOR_ASSET" ]; then
        # Asset exists
        if [ "$ALLOW_OVERWRITE" -eq 1 ]; then
            echo "  -> Overwriting '$FILENAME' in release '$RELEASE_FOR_ASSET'..."
            gh release upload "$RELEASE_FOR_ASSET" "$FILE_PATH" --clobber
        else
            echo "  -> Skipping '$FILENAME', it already exists in release '$RELEASE_FOR_ASSET'."
        fi
        continue
    fi

    # Find a release to upload to for a new asset
    if [ ${#AVAILABLE_RELEASES[@]} -eq 0 ]; then
        echo "Error: All existing releases are full. No space to upload '$FILENAME'." >&2
        exit 1
    fi

    UPLOAD_TARGET=${AVAILABLE_RELEASES[0]}

    echo "  -> Uploading '$FILENAME' to '$UPLOAD_TARGET'..."
    gh release upload "$UPLOAD_TARGET" "$FILE_PATH"

    # Update asset count and remove release from available list if full
    AVAILABLE_ASSET_COUNTS[0]=$((${AVAILABLE_ASSET_COUNTS[0]} + 1))
    
    MAX_ASSETS=998
    if [ "$UPLOAD_TARGET" == "$TAG" ]; then
        MAX_ASSETS=988
    fi

    if [ "${AVAILABLE_ASSET_COUNTS[0]}" -ge "$MAX_ASSETS" ]; then
        AVAILABLE_RELEASES=("${AVAILABLE_RELEASES[@]:1}")
        AVAILABLE_ASSET_COUNTS=("${AVAILABLE_ASSET_COUNTS[@]:1}")
    fi
done

echo "Upload process complete."
