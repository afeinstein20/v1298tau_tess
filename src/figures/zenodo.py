import requests
import os
import json
import sys


class ZenodoError(Exception):
    pass


def check_status(r):
    if r.status_code > 204:
        data = r.json()
        for error in data.get("errors", []):
            data["message"] += " " + error["message"]
        raise ZenodoError("Zenodo error {}: {}".format(data["status"], data["message"]))
    return r


def upload_simulation(
    file_name,
    deposit_title,
    deposit_description,
    access_token,
    sandbox=False,
    file_path=".",
):

    # Uplodad to sandbox (for testing) or to actual Zenodo?
    if sandbox:
        zenodo_url = "sandbox.zenodo.org"
    else:
        zenodo_url = "zenodo.org"

    # Search for an existing deposit with the given title
    print("Searching for existing deposit...")
    r = check_status(
        requests.get(
            f"https://{zenodo_url}/api/deposit/depositions",
            params={
                "q": deposit_title,
                "access_token": access_token,
            },
        )
    )
    deposit = None
    for entry in r.json():
        if entry["title"] == deposit_title:
            deposit = entry
            break

    # Either retrieve the deposit or create a new one
    if deposit:

        # Get the deposit id
        DEPOSIT_ID = deposit["id"]

        # Update the existing deposit
        print("Retrieving existing deposit...")
        try:

            # Create a new version draft
            r = check_status(
                requests.post(
                    f"https://{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}/actions/newversion",
                    params={"access_token": access_token},
                )
            )
            DEPOSIT_ID = r.json()["links"]["latest_draft"].split("/")[-1]

        except ZenodoError as e:

            if "403: Invalid action" in str(e):

                # Seems like we already have a draft. Let's use it
                DEPOSIT_ID = deposit["links"]["latest_draft"].split("/")[-1]

            else:

                raise e

        # Get the ID of the previously uploaded file (if it exists),
        # then delete it so we can upload a new version.
        print("Deleting old file...")
        r = check_status(
            requests.get(
                f"https://{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}/files",
                params={"access_token": access_token},
            )
        )
        for file in r.json():
            if file["filename"] == file_name:
                FILE_ID = file["id"]
                r = check_status(
                    requests.delete(
                        f"https://{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}/files/{FILE_ID}",
                        params={"access_token": access_token},
                    )
                )
                break

        # Upload the new version of the file
        print("Uploading new file...")
        with open(os.path.join(file_path, file_name), "rb") as fp:
            r = check_status(
                requests.post(
                    f"https://{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}/files",
                    data={"name": file_name},
                    files={"file": fp},
                    params={"access_token": access_token},
                )
            )

        # Publish the deposit
        print("Publishing the deposit...")
        try:
            r = check_status(
                requests.post(
                    f"https://{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}/actions/publish",
                    params={"access_token": access_token},
                )
            )
        except ZenodoError as e:
            if "New version's files must differ from all previous versions" in str(e):
                print("No change in the deposit's files. Aborting.")
            else:
                raise e

    else:

        # Create a new deposit
        print("Creating a new deposit...")
        r = check_status(
            requests.post(
                f"https://{zenodo_url}/api/deposit/depositions",
                params={"access_token": access_token},
                json={},
            )
        )

        # Get the deposit id
        DEPOSIT_ID = r.json()["id"]

        # Upload the file
        print("Uploading the file...")
        bucket_url = r.json()["links"]["bucket"]
        with open(os.path.join(file_path, file_name), "rb") as fp:
            r = check_status(
                requests.put(
                    "%s/%s" % (bucket_url, file_name),
                    data=fp,
                    params={"access_token": access_token},
                )
            )

        # Add some metadata
        print("Adding metadata...")
        data = {
            "metadata": {
                "title": deposit_title,
                "upload_type": "dataset",
                "description": deposit_description,
                "creators": [{"name": "Luger, Rodrigo"}],
            }
        }
        r = check_status(
            requests.put(
                f"https://{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}",
                params={"access_token": access_token},
                data=json.dumps(data),
                headers={"Content-Type": "application/json"},
            )
        )

        # Publish the deposit
        print("Publishing the deposit...")
        r = check_status(
            requests.post(
                f"https://{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}/actions/publish",
                params={"access_token": access_token},
            )
        )


def download_simulation(
    file_name,
    deposit_title,
    access_token,
    sandbox=False,
    file_path=".",
):

    # Uplodad to sandbox (for testing) or to actual Zenodo?
    if sandbox:
        zenodo_url = "sandbox.zenodo.org"
    else:
        zenodo_url = "zenodo.org"

    # Search for an existing deposit with the given title
    print("Searching for the deposit...")
    r = check_status(
        requests.get(
            f"https://{zenodo_url}/api/deposit/depositions",
            params={
                "q": deposit_title,
                "access_token": access_token,
            },
        )
    )
    deposit = None
    for entry in r.json():
        if entry["title"] == deposit_title:
            deposit = entry
            break

    if deposit is None:
        raise Exception("Cannot find deposit with the given title.")

    # Download the file
    print("Downloading file...")
    DEPOSIT_ID = deposit["id"]
    r = check_status(
        requests.get(
            f"https://{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}/files",
            params={"access_token": access_token},
        )
    )
    for file in r.json():
        if file["filename"] == file_name:
            FILE_ID = file["id"]
            r = check_status(
                requests.get(
                    f"https://{zenodo_url}/api/deposit/depositions/{DEPOSIT_ID}/files/{FILE_ID}",
                    params={"access_token": access_token},
                )
            )
            url = r.json()["links"]["download"]
            r = check_status(
                requests.get(
                    url,
                    params={"access_token": access_token},
                )
            )
            with open(file_name, "wb") as f:
                f.write(r.content)
            return

    raise Exception("Unable to download the file.")


# Name of the file to be uploaded
file_name = "simulation_results.dat"


# Name & description of the deposit on Zenodo
deposit_title = "Sample simulation results for showyourwork"
deposit_description = "A sample dataset uploaded using showyourwork."


# Zenodo access token. Create one here:
# https://zenodo.org/account/settings/applications/tokens/new/
# Remember to NEVER store it in unencrypted text files!
# I store it as an environment variable on my local machine, as well
# as a repository secret on GITHUB:
# https://docs.github.com/en/actions/security-guides/encrypted-secrets
access_token = os.getenv("ZENODO_TOKEN")


# Upload or download the file
if "--upload" in sys.argv:
    upload_simulation(file_name, deposit_title, deposit_description, access_token)
elif "--download" in sys.argv:
    download_simulation(file_name, deposit_title, access_token)
else:
    raise ValueError("Please specify either --upload or --download.")