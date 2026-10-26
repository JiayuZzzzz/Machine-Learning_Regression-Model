"""
submission_check.py  --  generic submission check library (student-facing)
==========================================================================
Handed out to students together with helpers.py.
Contains check functions and the run_checks() entry point.

Task-specific configuration (file paths, library versions, PDF fields,
feature count) lives in the term-specific script that students actually
run (aml2026_students_submission_check.py).

Do NOT run this file directly.
"""

import os
import re

import numpy as np
import sklearn

from helpers import load_dataset, load_model, read_pdf_fields


# -- Console output -----------------------------------------------------------

def _pass(msg): print(f"  [PASS] {msg}")
def _fail(msg): print(f"  [FAIL] {msg}")
def _info(msg): print(f"         {msg}")


def _section(title):
    print(f"\n{title}")


def _run(title, fn):
    """Print a section title, run fn(), catch and display any exception."""
    _section(title)
    try:
        fn()
    except Exception as exc:
        _fail(f"Unexpected error: {exc}")


# -- Check functions ----------------------------------------------------------

def check_library_version(lib_name, installed, expected):
    """Pass/fail for installed vs. expected library version."""
    if installed == expected:
        _pass(f"{lib_name} {installed}")
    else:
        _fail(
            f"{lib_name}: expected {expected}, got {installed} -- "
            "version mismatch may cause loading or prediction errors"
        )


def check_pickle_protocol(model_file, max_protocol):
    """Report the pickle protocol of the model file; fail if above max_protocol.

    Pickle files written with protocol N can only be loaded by Python versions
    that support protocol N.  Higher protocols are not backward-compatible.

    Protocol detection: protocols 2+ embed a PROTO opcode (0x80) as the first
    byte followed by the protocol number.  Protocols 0/1 have no magic header
    and are conservatively reported as protocol 1.

    Args:
        model_file   : path to the student's .pkl file.
        max_protocol : highest protocol version the grader's Python supports.
    """
    if not os.path.isfile(model_file):
        _fail(f"File not found: '{model_file}' -- cannot check protocol")
        return

    try:
        with open(model_file, "rb") as f:
            header = f.read(2)
    except Exception as exc:
        _fail(f"Could not read model file: {exc}")
        return

    if len(header) < 2:
        _fail("Model file is too short to determine pickle protocol")
        return

    if header[0] == 0x80:
        protocol = header[1]
    else:
        protocol = 1   # protocol 0 or 1 -- no PROTO opcode

    if protocol <= max_protocol:
        _pass(f"Pickle protocol {protocol} (max supported: {max_protocol})")
    else:
        _fail(
            f"Pickle protocol {protocol} exceeds maximum {max_protocol} -- "
            f"the grader's Python may not support this protocol; "
            f"re-save with:  pickle.dump(model, f, protocol={max_protocol})"
        )


def check_model_loadable(model_file):
    """Try to load model_file; print result and return model or None."""
    if not os.path.isfile(model_file):
        _fail(f"File not found: '{model_file}'")
        return None
    try:
        model = load_model(model_file)
    except TypeError as exc:
        _fail(f"Loaded object has no predict(): {exc}")
        return None
    except Exception as exc:
        _fail(f"Could not load model: {exc}")
        return None
    _pass(f"{type(model).__name__} loaded from '{model_file}'")
    return model


def check_predictions(model, data_file, n_features):
    """Run model.predict() on sample data; check shape, finiteness, diversity.

    Args:
        model      : fitted model with predict().
        data_file  : path to sample dataset CSV (header row expected).
        n_features : number of input features the model expects.
    """
    if not os.path.isfile(data_file):
        _fail(f"Data file not found: '{data_file}'")
        return

    try:
        X, y = load_dataset(data_file)
    except Exception as exc:
        _fail(f"Could not load dataset: {exc}")
        return

    if X.shape[1] != n_features:
        _fail(
            f"Dataset has {X.shape[1]} feature column(s), "
            f"expected {n_features}"
        )
        return

    try:
        preds = model.predict(X)
    except Exception as exc:
        _fail(f"predict() failed: {exc}")
        return

    preds = np.asarray(preds)

    if preds.shape != (len(y),):
        _fail(
            f"predict() returned shape {preds.shape}, "
            f"expected ({len(y)},)"
        )
        return
    _pass(f"predict() returned {preds.shape[0]} predictions")

    n_nan = int(np.sum(np.isnan(preds)))
    if n_nan:
        _fail(f"Predictions contain {n_nan} NaN value(s)")
        return
    _pass("No NaN values")

    n_inf = int(np.sum(np.isinf(preds)))
    if n_inf:
        _fail(f"Predictions contain {n_inf} Inf value(s)")
        return
    _pass("No Inf values")


def check_model_filename_match(pdf_file, pdf_fields, model_file):
    """Cross-check: model_filename declared in the PDF matches the actual .pkl.

    Args:
        pdf_file   : path to the student's PDF submission sheet.
        pdf_fields : {field_name: (type, display_label)} config dict.
        model_file : path to the student's model file.
    """
    if "model_filename" not in pdf_fields:
        return  # field not configured for this term; skip silently

    if not os.path.isfile(pdf_file):
        return  # step 4 already reported the missing PDF

    type_map = {k: t for k, (t, _) in pdf_fields.items()}
    try:
        values = read_pdf_fields(pdf_file, type_map)
    except Exception as exc:
        _fail(f"Could not read PDF for filename check: {exc}")
        return

    declared = (values.get("model_filename") or "").strip()
    actual = os.path.basename(model_file)

    if not declared:
        _fail(
            f"model_filename field is empty; "
            f"model file is '{actual}'"
        )
    elif declared != actual:
        _fail(
            f"PDF declares '{declared}' but model file is '{actual}'"
        )
    else:
        _pass(f"'{actual}' matches PDF declaration")


def check_pdf_sheet(filename, pdf_fields):
    """Check PDF filename pattern and report each AcroForm field individually.

    Args:
        filename   : path to the student's PDF submission sheet.
        pdf_fields : {field_name: (type, display_label)} for all fields.
    """
    if not os.path.isfile(filename):
        _fail(f"PDF file not found: '{filename}'")
        return

    # Filename convention check
    basename = os.path.basename(filename)
    if re.fullmatch(r"[A-Z]{2}_\d{3}\.pdf", basename):
        _pass(f"Filename '{basename}' matches required pattern")
    else:
        _fail(
            f"Filename '{basename}' does not match pattern "
            "<INITIALS>_<last-3-digits>.pdf  (e.g. MS_042.pdf)"
        )

    # Read all PDF fields
    try:
        type_map = {k: t for k, (t, _) in pdf_fields.items()}
        values = read_pdf_fields(filename, type_map)
    except Exception as exc:
        _fail(f"Could not read PDF fields: {exc}")
        return

    # Report each field individually
    for key, (dtype, display) in pdf_fields.items():
        val = values[key]
        if val is None or (isinstance(val, str) and not val.strip()):
            _fail(f"{display}: empty or not readable as {dtype.__name__}")
        else:
            _pass(f"{display}: {val!r}")


# -- Entry point --------------------------------------------------------------

def run_checks(config):
    """Run all submission checks from config and print results.

    Expected config keys
    --------------------
    model_file           str   path to the student's .pkl
    data_file            str   path to the sample dataset
    pdf_file             str   path to the student's submission PDF
    n_features           int   number of input features the model expects
    pdf_fields           dict  {field_name: (type, display_label), ...}
    sklearn_version      str   required sklearn version string
    numpy_version        str   required numpy version string
    pickle_protocol_max  int   highest pickle protocol the grader supports
    """
    print("=" * 54)
    print("  Submission Check")
    print("=" * 54)

    _run("1. Library versions", lambda: (
        check_library_version(
            "sklearn", sklearn.__version__, config["sklearn_version"]
        ),
        check_library_version(
            "numpy", np.__version__, config["numpy_version"]
        ),
    ))

    _run("2. Pickle protocol", lambda: check_pickle_protocol(
        config["model_file"], config["pickle_protocol_max"]
    ))

    model = None

    def _load():
        nonlocal model
        model = check_model_loadable(config["model_file"])

    _run("3. Model loading", _load)

    if model is not None:
        _run("4. Predictions on sample data", lambda: check_predictions(
            model, config["data_file"], config["n_features"]
        ))
    else:
        _section("4. Predictions on sample data")
        _fail("Skipped -- model could not be loaded")

    _run("5. PDF submission sheet", lambda: check_pdf_sheet(
        config["pdf_file"], config["pdf_fields"]
    ))

    _run("6. Model filename in PDF", lambda: check_model_filename_match(
        config["pdf_file"], config["pdf_fields"], config["model_file"]
    ))

    print("\n" + "=" * 54)
