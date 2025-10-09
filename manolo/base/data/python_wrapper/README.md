# Manolod Data Tier Python Client:

A Python wrapper for the **Manolo Data Tier API**.

This project provides an easy-to-use interface to interact with the Manolo service.

---

## Installation

1. **Clone the repository:**

   git clone https://github.com/data-eng/manolo-wp2.git

   cd manolo-wp2

(Optional). Create a Virtual Environment

**Using Python `venv`**

    python -m venv venv

Activate the Enviroment

_Linux/macOS_

    source venv/bin/activate

_Windows_

    venv\Scripts\activate

**Using Conda**

    conda create -n manolo-client python=3.9

    conda activate manolo-client

2.  **Install dependencies:**

        pip install -r requirements.txt

---

## Usage

1.  **Use the python client**

    For more details on how to use it check the `demo/` folder.

## Notes

1.  **In case the user tries to upload something after it was uploaded in another way:**

        Uploaded Encrypted -> Tries to upload Unencrypted

        Uploaded Unencrypted -> Tries to upload Encrypted

The system will not upload the item again (unless it has changes since last time).

---

## Concepts

### DSN (Data Structure Number)

A **DSN** is the unique identifier of a data structure in the Data Tier.
It is per data structure for example the application by default makes 2 data structures:

        mlflow with the DSN 1
        topology with the DSN 2

The DSN is similar to anyone that is connected to the same instance of the Data Tier Service.

### Manifest file

The **manifest.json** is created by the service and is exclusively used when uploading encrypted items.

### Regarding the unique keys:

    The long 29-character value is the internal database ID (not user-friendly).

    The DSN is a number that should carry meaning for the user who created the data structure, so it must be unique.
    
    The Name is intended as a more human-readable identifier, easier to recall than the DSN.

This way, users have flexibility: a unique DSN and Name for structure identification, plus a free-form Description for convenience.
