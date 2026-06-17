"""Shared helpers for generated-code execution and LLM-assisted repair."""

import importlib
import json
import multiprocessing as mp
import os
import queue as queue_module
import re
import shutil
import textwrap
import traceback
from datetime import datetime

import tools.Code_Exec as Code_Exec
import tools.VisualQuestion as VQ


def sanitize_code(code_str: str) -> str:
    """Normalize punctuation that LLMs often emit inside Python code blocks."""
    # Replace smart punctuation with plain Python-safe characters.
    for old, new in [('“', '"'), ('”', '"'), ('’', "'"), ('‘', "'"), ('—', '-'), ('–', '-')]:
        code_str = code_str.replace(old, new)
    return code_str.strip()


def check_and_install_dependencies(code, model="", num_tries=3, messages=None):
    """Ask the LLM for dependency setup code and run it before validation."""
    # Build the dependency-analysis prompt when no conversation was supplied.
    if messages is None:
        prompt = (
            "Read the following code and identify its packages, imports, and dependencies.\n"
            "Write a Python script that checks whether those dependencies are available "
            "and installs them if necessary.\n\n"
            "Return JSON in this format: {'packages': list of required packages, or "
            "'None' if there are none, 'installation_code': Python code that checks and "
            "installs the packages. If no installation is needed, leave it empty.}"
        )
        messages = [
            {"role": "system", "content": "You are a software developer."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Here is the code:\n\n" + code},
        ]
    print(messages)

    # Let the LLM propose and repair dependency installation code.
    for i in range(num_tries):
        results = VQ.get_reponse(messages=messages, model=model, as_json=True)
        messages.append({"role": "system", "content": str(results)})
        print(messages[-1])

        # Ask again if the response is not structured JSON.
        if not isinstance(results, dict):
            messages.append(
                {
                    "role": "user",
                    "content": "Return a valid JSON object with the requested fields.",
                }
            )
            continue

        # First pass expects package names and install/check code.
        if i == 0:
            packages = results.get("packages")
            installation_code = results.get("installation_code", "")
            if packages in (None, [], "none", "None") or len(installation_code) == 0:
                return True, messages, ""
            code_to_run = installation_code
        else:
            # Later passes expect fixed install code after a failure.
            if "yes" not in str(results.get("solvable", "")).lower():
                return False, messages, ""
            code_to_run = results.get("fixed_code", "")
            if len(code_to_run) == 0:
                return False, messages, ""

        # Run the proposed dependency installer.
        print("Trying to install dependencies using:\n\n", textwrap.dedent(code_to_run))
        succeeded, captured_stdout, captured_stderr = Code_Exec.run_code(
            textwrap.dedent(code_to_run)
        )
        if succeeded:
            return True, messages, code_to_run

        # Feed installer errors back into the LLM for another repair attempt.
        messages.append(
            {
                "role": "user",
                "content": (
                    "Dependency installation failed with this error:\n"
                    + captured_stderr
                    + "\n\nTry to solve the error. Return JSON in this format: "
                    "{'packages': list of packages to install or 'None', "
                    "'solvable': 'yes' or 'no', 'fixed_code': code ready to run}"
                ),
            }
        )
        print(messages[-2:])

    return False, messages, ""


def _run_test_function(queue, test_function, code_dir, num_samples, texture_size, output_dir):
    """Process target used by run_debug_code so timeouts can stop generated code."""
    try:
        # Run generated code and send success back to the parent process.
        result = test_function(code_dir, num_samples, texture_size, output_dir)
        queue.put({"ok": True, "result": result})
    except Exception:
        # Return the full traceback so the repair model has useful context.
        queue.put({"ok": False, "error": traceback.format_exc()})


def _multiprocessing_context():
    """Choose a multiprocessing context that works for generated-code execution."""
    # Prefer fork on Unix so local test functions do not need pickling.
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def run_debug_code(
    messages,
    code,
    code_dir,
    codefilename,
    task_description,
    test_function,
    num_iter=4,
    clean_dir=True,
    time_out=0,
    rechek_code=False,
    model="",
    pre_install_dependency=False,
    num_samples=10,
    texture_size=512,
    output_dir=None,
):
    """Write generated code, execute it, and ask the LLM to repair crashes."""
    # Track the final status and path across repair attempts.
    code_verified = False
    path = os.path.join(code_dir, codefilename)

    # Optionally install dependencies before running generated code.
    if pre_install_dependency:
        inst_success, inst_logs, installation_code = check_and_install_dependencies(
            code, model=model
        )
        messages += inst_logs
        if not inst_success:
            return False, path, code_dir, code, "", messages

    # Try the original code and any LLM repairs.
    for ii in range(num_iter):
        # Clean previous generated files when requested.
        if os.path.exists(code_dir) and clean_dir:
            shutil.rmtree(code_dir)
        os.makedirs(code_dir, exist_ok=True)

        # Write the generated module that the tester will import.
        path = os.path.join(code_dir, codefilename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        importlib.invalidate_caches()

        print(
            "\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n"
            + "Running the code in "
            + path
            + "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4\n\n"
        )

        out_text = ""
        captured_err = ""
        is_error = False

        # Resolve runtime settings for this validation attempt.
        timeout_seconds = time_out if time_out > 0 else 5000
        test_output_dir = output_dir or os.path.join(code_dir, "generated_textures512")
        print(
            "Testing code "
            + code_dir
            + "\nTime out "
            + str(timeout_seconds)
            + " seconds.\nStart time:"
            + datetime.now().strftime("%H:%M:%S")
        )

        # Run generated code in a child process so timeouts can kill it.
        context = _multiprocessing_context()
        queue = context.Queue()
        process = context.Process(
            target=_run_test_function,
            args=(queue, test_function, code_dir, num_samples, texture_size, test_output_dir),
        )
        process.start()
        process.join(timeout_seconds)

        # Hard-stop code that runs past the configured timeout.
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            captured_err = "\nThe code takes too long to run; make it more efficient.\n"
            is_error = True
        else:
            # Read the child process result or crash payload.
            try:
                result_payload = queue.get(timeout=1)
            except queue_module.Empty:
                result_payload = None

            # Interpret the child process result.
            if result_payload and result_payload.get("ok"):
                print("Result:", result_payload.get("result"))
            else:
                is_error = True
                print("The program crashed")
                if result_payload:
                    captured_err = "The code crashed with error:\n" + result_payload.get(
                        "error", ""
                    )
                else:
                    captured_err = (
                        "The code process exited with status "
                        + str(process.exitcode)
                        + " and no error payload."
                    )

        # Close process communication resources.
        queue.close()
        queue.join_thread()

        # Ask the LLM to repair code that crashed or timed out.
        if is_error:
            print(
                "\niiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n\n"
                "CODE running failed\n"
                "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n\n"
            )
            text = (
                "***Code***\n"
                + code
                + "\n\n***Error***\n"
                + captured_err
                + "\n\nAnalyze the code and fix it if possible. Return JSON with these fields:\n"
                + "{'fixable': 'yes' or 'no', 'code': fixed clean code, "
                + "'details': describe the errors and changes, "
                + "'dependencies': 'yes' or 'no' if new or reinstalled dependencies are needed}"
            )
            messages.append({"role": "user", "content": text})
            print(messages[-1])
            results = VQ.get_reponse(messages=messages, model=model, as_json=True)

            messages.append({"role": "system", "content": str(results)})
            print(messages[-1])

            # Continue with repaired code when the LLM supplies it.
            if isinstance(results, dict) and str(results.get("fixable", "")).lower() == "yes":
                fixed_code = results.get("code", "")
                if len(fixed_code) == 0:
                    break
                code = sanitize_code(fixed_code)

                # Install any new dependencies requested by the repair.
                if str(results.get("dependencies", "")).lower() == "yes":
                    inst_success, inst_logs, installation_code = check_and_install_dependencies(
                        code, model=model
                    )
                    messages += inst_logs
                continue
            break

        # Generated code ran without crashing.
        print(
            "\n\nVVVVVVVVVVVVVVVVVVVVVVVVVV\n"
            "CODE running Succeed\n"
            "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV\n\n"
        )

        # Skip expensive semantic recheck unless requested.
        if task_description == "" or not rechek_code:
            code_verified = True
            break

        # Ask the LLM to inspect successful code for task-level mistakes.
        text = (
            "***Analyze the following code***:\n"
            + code
            + "\n\n***The code ran successfully and output:***\n"
            + out_text
            + "\n\n***Task description***\n\n"
            + task_description
            + "\n\n***Go over the code, task description, and output. Report only errors "
            + "in the generated texture code. Return JSON with these fields: "
            + "{'error': 'yes' or 'no', 'fixable': 'yes' or 'no', "
            + "'code': fixed code ready to run, 'description': description of the error found}***"
        )
        messages.append({"role": "user", "content": text})
        print(messages[-1])

        results = VQ.get_response_image_txt_json(text=text, model=model)
        messages.append({"role": "system", "content": str(results)})
        print(messages[-1])

        # Accept code that the semantic checker approves.
        if isinstance(results, dict) and str(results.get("error", "")).lower() == "no":
            code_verified = True
            break

        # Otherwise continue with checker-provided repaired code.
        if isinstance(results, dict) and str(results.get("fixable", "")).lower() == "yes":
            fixed_code = results.get("code", "")
            if len(fixed_code) == 0:
                break
            code = sanitize_code(fixed_code)
        else:
            break

    # Ensure the code directory exists before writing final status files.
    os.makedirs(code_dir, exist_ok=True)

    # Store the debug conversation for later inspection.
    with open(os.path.join(code_dir, "Testing_logs.json"), "w", encoding="utf-8") as fl:
        json.dump(messages, fl, indent=4)

    # Mark that the debug loop finished.
    with open(os.path.join(code_dir, "finish.txt"), "w", encoding="utf-8") as fl:
        fl.write("Finished")

    # Mark verified code for file-system-level inspection.
    if code_verified:
        with open(os.path.join(code_dir, "verified.txt"), "w", encoding="utf-8") as fl:
            fl.write("Verified")

    return code_verified, path, code_dir, code, out_text, messages


def path_to_import(path: str, base: str = None) -> str:
    """Convert a filesystem path to a dotted import path."""
    # Normalize separators and remove the file extension.
    module_path = os.path.normpath(path)
    module_path = os.path.splitext(module_path)[0]

    # Remove an optional base prefix before splitting into module parts.
    if base and module_path.startswith(base):
        module_path = module_path[len(base):]

    # Convert every path component into a valid Python identifier.
    clean_parts = []
    for part in module_path.strip(os.sep).split(os.sep):
        identifier = re.sub(r"[^0-9a-zA-Z_]", "_", part)
        if not identifier:
            continue
        if identifier[0].isdigit():
            identifier = "_" + identifier
        clean_parts.append(identifier)

    return ".".join(clean_parts)
