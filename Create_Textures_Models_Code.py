"""Generate PBR texture models with an LLM-assisted pipeline.

The pipeline has three stages:
1. Ask an LLM to suggest scientific or procedural texture-generation methods.
2. Ask an LLM to implement each method as a Python PBR texture generator.
3. Execute, validate, and optionally repair each generated script.
"""

from datetime import datetime
import os
import pickle
import random
import re

import check_pbr_errors
import code_tester
import tools.MainFunctions as MF
import tools.VisualQuestion as VQ


def _load_query_prompts(query_dir):
    """Load all prompt files from a query directory."""
    prompts = {}

    # Read each prompt file into the query dictionary.
    for filename in os.listdir(query_dir):
        path = os.path.join(query_dir, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as prompt_file:
            prompts[filename] = prompt_file.read()
    return prompts


def _save_dataset_state(data, data_file, backup_file=None):
    """Persist the dataset state and optional backup copy."""
    # Write the primary dataset state.
    with open(data_file, "wb") as fl:
        pickle.dump(data, fl)

    # Keep a backup when the caller requests one.
    if backup_file is not None:
        with open(backup_file, "wb") as fl:
            pickle.dump(data, fl)


def _benchmark_dir_name(name):
    """Convert a benchmark name into the directory-safe name used by the dataset."""
    # Keep only letters so the generated folder is import-safe enough.
    safe_name = re.sub(r"[^a-zA-Z]", "_", name).strip("_")
    return safe_name or "benchmark"


def generate_dataset(
    dataset_dir,
    query_dir,
    number_of_new=10,
    number_of_code_fix_retry=2,
    recheck_originality=True,
    idea_model="",
    code_model="",
    check_model="",
    model="",
):
    """Generate or update  dataset directory.

    ``recheck_originality`` is kept for compatibility with older callers; the
    current pipeline relies on the suggestion prompt plus loaded dataset history
    to avoid repeated methods.
    """
    # Use the fallback model for any role-specific model not provided.
    if idea_model == "":
        idea_model = model
    if code_model == "":
        code_model = model
    if check_model == "":
        check_model = model

    # Prepare dataset paths.
    data_file = os.path.join(dataset_dir, "data.pkl")
    data_file_back = os.path.join(dataset_dir, "data_back.pkl")
    os.makedirs(dataset_dir, exist_ok=True)

    # Start from fresh prompts; persisted data will be merged below.
    dt = {"qr": _load_query_prompts(query_dir), "messages": []}

    # Resume an existing dataset if it already has saved state.
    data_file_loaded = False
    if os.path.isfile(data_file):
        print("\n\n\nLoad file\n\n\n")
        query_prompts = dt["qr"]
        with open(data_file, "rb") as fl:
            dt = pickle.load(fl)

        # Always use the current prompt files, not stale pickled prompts.
        dt["qr"] = query_prompts
        dt.setdefault("messages", [])
        if not isinstance(dt.get("benchmarks"), dict):
            dt["benchmarks"] = {}
        data_file_loaded = True

        # Drop malformed or unused benchmark entries from older runs.
        for bname, ent in list(dt["benchmarks"].items()):
            remove_entry = False
            if not isinstance(ent, dict) or "description" not in ent:
                remove_entry = True
            elif ent.get("full_overlap") is True and "code" not in ent:
                remove_entry = True

            if remove_entry:
                print("Removing:", bname)
                del dt["benchmarks"][bname]

    # Ask the idea model for new scientific texture methods.
    if number_of_new > 0:
        print("\n\n\nSuggest Model for Patterns and Texture generations\n\n\n")
        txt = dt["qr"]["suggest_benchmarks"].replace(
            "@@@number_of_new@@@", str(number_of_new)
        )

        # Include prior method names to reduce duplicate suggestions.
        if data_file_loaded:
            existing_methods = ", ".join(str(ky) for ky in dt["benchmarks"])
            txt += (
                dt["qr"]["add_suggestions"]
                + "\nPrevious suggested methods: ["
                + existing_methods
                + "]"
            )

        resp, dt = VQ.get_reponse(dt, text=txt, model=idea_model)

        # Convert the free-form suggestions into benchmark JSON.
        dt["benchmarks_text"] = resp
        dt["messages"].append({"role": "user", "content": dt["qr"]["suggestions_to_json"]})
        benchmark_dic = VQ.get_reponse(
            messages=dt["messages"][-3:], as_json=True, model=idea_model
        )
        dt["messages"].append({"role": "system", "content": str(benchmark_dic)})

        # Merge only new benchmark entries into the dataset.
        if not isinstance(benchmark_dic, dict):
            print("Suggestion JSON conversion failed; no benchmarks were added.")
        elif "benchmarks" in dt:
            for ky, benchmark in benchmark_dic.items():
                if ky not in dt["benchmarks"]:
                    dt["benchmarks"][ky] = benchmark
                    print(benchmark)
                else:
                    print("error", ky, "already exists")
        else:
            dt["benchmarks"] = benchmark_dic

        _save_dataset_state(dt, data_file)

    # Process each benchmark independently.
    dt.setdefault("benchmarks", {})
    for bname in list(dt["benchmarks"]):
        print("benchmark", bname)
        ent = dt["benchmarks"][bname]

        # Skip entries that cannot drive code generation.
        if not isinstance(ent, dict) or "description" not in ent:
            print("Skipping malformed benchmark:", bname)
            continue

        bdesc = ent["description"]
        code_info = ent.get("code")

        already_done = (
            isinstance(code_info, dict)
            and (code_info.get("code verified") is True or code_info.get("finished_and_failed"))
        )

        # Do not regenerate completed, failed, or duplicate methods.
        if already_done:
            continue
        if ent.get("full_overlap"):
            continue

        # Build the implementation prompt for this benchmark.
        code_query = (
            dt["qr"]["implement_code"]
            .replace("**method_name**", bname)
            .replace("**method_description**", bdesc)
            .replace(
                "Plus not less then 2 of (more is better)",
                "In addition to at least 2 or more of the maps listed below:",
            )
            + "\nSome maps can be uniform, but at least 3 maps should be textured."
        )

        # Ask the coding model to produce a generator if none exists.
        if not isinstance(code_info, dict) or "Succeed" not in code_info:
            print("\n\n\nWrite code for:" + bname + "\n\n\n")
            for _ in range(10):
                print(code_query)
                code_dic, dt = VQ.get_reponse(
                    dt, text=code_query, as_json=True, model=code_model
                )

                # Retry if the model did not return a usable JSON object.
                if not isinstance(code_dic, dict):
                    continue

                # Accept older responses that include code but omit Succeed.
                if (
                    "code" in code_dic
                    and "Succeed" not in code_dic
                    and len(str(code_dic["code"])) > 100
                ):
                    code_dic["Succeed"] = "yes"
                if "Succeed" not in code_dic:
                    continue

                # Store the first usable code payload.
                ent["code"] = code_dic
                ent["code"]["query"] = code_query
                break

            _save_dataset_state(dt, data_file)

        # Re-read code info after possible generation.
        code_info = ent.get("code")
        if not isinstance(code_info, dict):
            print("No usable code payload for:", bname)
            continue
        if str(code_info.get("Succeed", "")).lower() == "no" or "code" not in code_info:
            continue

        # Execute and validate any code that has not been verified yet.
        if code_info.get("code verified") is not True:
            print("\n\n\nTest and validate code for:" + bname + "\n\n\n")

            # Create stable output paths for this benchmark.
            sname = _benchmark_dir_name(bname)
            ent["simple name"] = sname
            outcodedir = os.path.join(dataset_dir, sname)
            sample_dir = os.path.join(outcodedir, "textures")
            ent["dir"] = outcodedir

            # Validation settings expected by the generated function.
            texture_size = 512
            numsamples = 10
            max_repair_rounds = max(number_of_code_fix_retry, 0)
            code_verified = False
            code = code_info["code"]

            # Run the generator and optionally retry repairs.
            for kk in range(max_repair_rounds + 100):
                os.makedirs(outcodedir, exist_ok=True)
                code = ent["code"]["code"]
                message_count_before_test = len(dt["messages"])

                # Execute generated code inside the shared debug helper.
                code_verified, path, test_dir, code, captured_stdout, messages = MF.run_debug_code(
                    messages=dt["messages"][-2:],
                    code=code,
                    code_dir=outcodedir,
                    codefilename="generate.py",
                    task_description=bdesc,
                    time_out=3000,
                    rechek_code=False,
                    model=code_model,
                    test_function=code_tester.run,
                    num_samples=numsamples,
                    texture_size=texture_size,
                    output_dir=sample_dir,
                )

                # Save human-readable metadata beside the generated script.
                with open(
                    os.path.join(test_dir, "Description.txt"), "w", encoding="utf-8"
                ) as fl:
                    fl.write(bdesc)
                if "overlap" in ent:
                    with open(os.path.join(test_dir, "overlap.txt"), "wb") as fl:
                        pickle.dump(ent["overlap"], fl)

                # Stop if the code cannot run.
                if not code_verified:
                    dt["messages"] = dt["messages"][:message_count_before_test]
                    break

                # Check folder structure, map names, image sizes, and diversity.
                find_error, error_message = check_pbr_errors.verify_pbr_output(
                    outdir=sample_dir,
                    numsamples=numsamples,
                    sz=texture_size,
                )

                # Build a checker prompt containing the code and test result.
                txt = "\n\n\n***Your task:***\n" + dt["qr"]["check_code"]
                txt += "***Original Code Generation task:***\n" + code_query + "\n\n\n"
                txt += "***Generated Code***:\n" + code + "\n\n\n"
                txt += "The code ran on time and did not crash.\n"
                if find_error:
                    txt += (
                        "However, automatic inspection found these output problems:\n"
                        + error_message
                    )
                else:
                    txt += (
                        "Automatic inspection: file structure, map names, and image sizes "
                        "are correct.\n"
                    )

                # Accept the result once retry limits are reached.
                if kk >= max_repair_rounds and not find_error:
                    break
                if kk >= max_repair_rounds + 3:
                    break

                # Ask the checker model for a corrected generator.
                verify_query = txt
                print(txt)
                code_dic, dt = VQ.get_reponse(
                    dt, text=verify_query, as_json=True, model=check_model
                )
                if (
                    isinstance(code_dic, dict)
                    and str(code_dic.get("corrections", "")).lower() == "yes"
                    and "code" in code_dic
                ):
                    # Replace the current code with the corrected version.
                    print(
                        "\n\n\n****************************************************************************\n\n\n"
                        "Code correction found:",
                        code_dic,
                    )
                    ent["code"] = code_dic
                    ent["code"]["Succeed"] = "yes"
                    ent["code"]["query"] = code_query
                    ent["code"]["fixing_query"] = verify_query
                else:
                    break

            # Store the final code and verification result.
            ent["code"]["code"] = code
            ent["code"]["code verified"] = code_verified
            if not code_verified:
                ent["code"]["finished_and_failed"] = True

            # Persist after each benchmark so long runs can resume safely.
            _save_dataset_state(dt, data_file, data_file_back)
            print(
                "\n\n\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\n"
                "Finished benchmark ",
                bname,
                "\nverified ",
                code_verified,
                "\n\n\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\n",
            )


if __name__ == "__main__":
    # Models used for new idea suggestions.
    idea_models = [
        "google/gemini-3.1-flash-lite-preview",
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "openai/gpt-5.5",
        "openai/gpt-5.4",
        "qwen/qwen3.6-max-preview",
        "xiaomi/mimo-v2.5-pro",
        "deepseek/deepseek-v4-pro",
        "google/gemini-3.1-pro-preview",
        "minimax/minimax-m2.7",
        "anthropic/claude-sonnet-4.6",
        "x-ai/grok-4.3",
    ]

    # Models used for code generation and repair.
    coding_models = [
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.2",
        "openai/gpt-5.5",
        "openai/gpt-5.4",
    ]

    # Dataset groups based on model type to sample during the long-running loop.
    main_outdir = "output_pbrs"
    queries = [
        "pbr_complex_original",
        "pbr_social_science",
        "pbr_biology",
        "pbr_phsyics_engineering",
        "pbr_creative",
        "pbr_math",
    ]
    print(queries)
    os.makedirs(main_outdir, exist_ok=True)

    # Continuously expand random dataset groups.
    for _ in range(1000):
        topic = random.choice(queries)
        query_dir = os.path.join("queries_prompts", topic)
        outputdir = os.path.join(main_outdir, topic)
        os.makedirs(outputdir, exist_ok=True)

        # Runtime knobs for this generation round.
        number_of_code_fix_retry = 0
        recheck_originality = False
        model = random.choice(coding_models)
        idea_model = random.choice(idea_models)
        number_of_new = 5

        print(datetime.now())
        print(
            "\n*************************\nIdea model:"
            + idea_model
            + "\nCoding model:"
            + model
            + "\nQuery dir:"
            + query_dir
            + "\n********************************\n"
        )

        # Generate new methods and validate pending code in this group.
        generate_dataset(
            dataset_dir=outputdir,
            query_dir=query_dir,
            number_of_new=number_of_new,
            number_of_code_fix_retry=number_of_code_fix_retry,
            recheck_originality=recheck_originality,
            model=model,
            idea_model=idea_model,
            code_model=model,
        )
