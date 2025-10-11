import numpy as np
import logging
import os
import time
import json
from llm_client import PromptManager
from tqdm.auto import tqdm
from utils import clean_response, alphabetize_csv_text, get_csv_text_n_entries, load_required_files, save_memory

logger = logging.getLogger(__name__)


def _generate_with_prompts(llm_client, prompts, kwargs_list, do_sleep_flags=None):
    """Helper to generate responses from multiple prompts sequentially."""
    if do_sleep_flags is None:
        do_sleep_flags = [True] * len(prompts)
    
    responses = []
    for i, (prompt_key, kwargs) in enumerate(zip(prompts.keys(), kwargs_list)):
        prompt = PromptManager.format_prompt(prompts[prompt_key], **kwargs)
        logger.info(f"Prompt {i+1}: {prompt}")
        
        full_response, extracted = llm_client.generate_and_extract(
            prompt, do_sleep=do_sleep_flags[i] if i < len(do_sleep_flags) else False
        )
        responses.append((full_response, extracted))
    
    return responses


def run_phonology_step(args, llm_client):
    """Run the phonology generation step."""
    # Load prompts
    prompt_dir = os.path.join(args.prompt_dir, 'phonology')
    prompts = PromptManager.load_prompts(
        prompt_dir,
        ['phon_step1_checklist.txt', 'phon_step2_summary.txt', 'phon_step3_word_shapes.txt']
    )
    
    custom = "(none)" if args.custom_constraints is None else args.custom_constraints
    values = np.random.randint(args.phon_n_answers, size=args.phon_n_questions) + 1
    
    # Prepare all kwargs
    kwargs_list = [
        {
            'n_questions': args.phon_n_questions,
            'n_answers': args.phon_n_answers,
            'scale_size': args.phon_scale_size
        }
    ]
    
    # Generate step 1
    responses = _generate_with_prompts(llm_client, {'step1': prompts['phon_step1_checklist']}, kwargs_list)
    _, answer_checklist = responses[0]
    
    # Steps 2 and 3
    remaining_kwargs = [
        {
            'checklist': answer_checklist,
            'values': str(list(values)),
            'custom': custom
        },
        {
            'phonology': None,  # Will be set after step 2
            'n': args.phon_n_words,
            'custom': custom
        }
    ]
    
    # Step 2
    step2_prompts = {'step2': prompts['phon_step2_summary']}
    step2_responses = _generate_with_prompts(llm_client, step2_prompts, [remaining_kwargs[0]], [False])
    _, answer_phonology = step2_responses[0]
    
    # Step 3
    remaining_kwargs[1]['phonology'] = answer_phonology
    step3_prompts = {'step3': prompts['phon_step3_word_shapes']}
    step3_responses = _generate_with_prompts(llm_client, step3_prompts, [remaining_kwargs[1]], [False])
    _, answer_word_shapes = step3_responses[0]

    # Concatenate to full phonology to save out
    answer_phonology_full = answer_phonology.strip() + '\n\n' + answer_word_shapes.strip()
    
    logger.info(f"Saving phonology summary in memory")
    metadata = {**kwargs_list[0], **remaining_kwargs[0], **remaining_kwargs[1]}
    
    # Create step-specific directory
    step_memory_dir = os.path.join(args.memory_dir, 'phonology')
    save_memory(answer_phonology_full, step_memory_dir, 'phonology.txt', metadata)
    
    logger.info("Phonology step completed")
    return True


def run_grammar_step(args, llm_client):
    """Run the grammar generation step."""
    # Load existing data
    files = load_required_files(args.memory_dir, {'phonology': 'phonology.txt'})
    if files is None:
        return False
    phonology = files['phonology']
    
    # Load prompts
    prompt_dir = os.path.join(args.prompt_dir, 'grammar')
    prompts = PromptManager.load_prompts(
        prompt_dir,
        ['gram_step1_checklist.txt', 'gram_step2_summary.txt',
        'gram_step3_expand.txt', 'merge_sections.txt']
    )
    
    custom = "(none)" if args.custom_constraints is None else args.custom_constraints
    values = np.random.randint(args.gram_n_answers, size=args.gram_n_questions) + 1
    
    # Generate all steps
    kwargs_list = [
        {
            'n_questions': args.gram_n_questions,
            'n_answers': args.gram_n_answers,
            'scale_size': args.gram_scale_size
        }
    ]
    
    # Step 1
    step1_responses = _generate_with_prompts(
        llm_client, {'step1': prompts['gram_step1_checklist']}, kwargs_list
    )
    _, answer_checklist = step1_responses[0]
    
    # Steps 2-4
    step2_kwargs = {
        'checklist': answer_checklist,
        'values': str(list(values)),
        'custom': custom,
        'phonology': phonology,
    }
    
    step2_responses = _generate_with_prompts(
        llm_client, {'step2': prompts['gram_step2_summary']}, [step2_kwargs]
    )
    _, grammar = step2_responses[0]
    
    step3_kwargs = {
        'grammar': grammar,
        'custom': custom,
        'phonology': phonology,
    }
    
    step3_responses = _generate_with_prompts(
        llm_client, {'step3': prompts['gram_step3_expand']}, [step3_kwargs]
    )
    _, expanded_grammar = step3_responses[0]
    
    # Step 4: Merge sections
    summaries = f"===SUMMARY 1:===\n{grammar}\n\n===SUMMARY 2:===\n{expanded_grammar}\n===END SUMMARIES==="
    step4_kwargs = {'summaries': summaries}
    
    step4_responses = _generate_with_prompts(
        llm_client, {'step4': prompts['merge_sections']}, [step4_kwargs], [False]
    )
    _, merged_grammar = step4_responses[0]
    
    # Save results
    logger.info(f"Saving grammar summary in memory")
    metadata = {**kwargs_list[0], **step2_kwargs, **step3_kwargs}
    
    step_memory_dir = os.path.join(args.memory_dir, 'grammar')
    save_memory(merged_grammar, step_memory_dir, 'grammar.txt', metadata)

    logger.info("Grammar step completed")
    return True


def _run_iterative_csv_step(args, llm_client, step_name, required_files, prompt_files, 
                           min_entries_attr, n_per_iter_attr, extra_sleep_attr, max_iters_attr):
    """Generic function for iterative CSV generation steps (lexicon, corpus)."""
    
    # Load existing data
    files = load_required_files(args.memory_dir, required_files)
    if files is None:
        return False
    
    # Load prompts
    prompt_dir = os.path.join(args.prompt_dir, step_name)
    loaded_prompts = PromptManager.load_prompts(prompt_dir, prompt_files)
    
    # Get prompt keys (filenames without extension)
    prompt1_key = prompt_files[0].replace('.txt', '')
    prompt2_key = prompt_files[1].replace('.txt', '')

    min_entries = getattr(args, min_entries_attr)
    n_per_iter = getattr(args, n_per_iter_attr)
    extra_sleep = getattr(args, extra_sleep_attr)
    max_iters = getattr(args, max_iters_attr)
    
    pbar = tqdm(desc=f"Making {step_name}", total=min_entries)

    # Step 1: Extract existing items
    step1_kwargs = {k: v for k, v in files.items() if k in ['phonology', 'grammar']}
    
    step1_responses = _generate_with_prompts(
        llm_client, {'step1': loaded_prompts[prompt1_key]}, [step1_kwargs]
    )
    _, csv_raw = step1_responses[0]
    
    csv_data = clean_response(csv_raw, "csv")
    csv_data = alphabetize_csv_text(csv_data)

    n_entries = get_csv_text_n_entries(csv_data)
    pbar.update(n_entries)

    # Step 2: Expand with more items
    i = 0
    while get_csv_text_n_entries(csv_data) < min_entries and i < max_iters:
        # Add extra sleep time
        logger.info(f"Extra sleep for {step_name}: sleeping for {extra_sleep} seconds...")
        if not args.debug:
            time.sleep(extra_sleep)

        i += 1
        pbar.set_description(f"Making {step_name} (expansion iter {i}/{max_iters})")
    
        step2_kwargs = {**files, step_name: csv_data, 'n': n_per_iter}
        
        step2_responses = _generate_with_prompts(
            llm_client, {'step2': loaded_prompts[prompt2_key]}, [step2_kwargs]
        )
        _, expanded_csv_raw = step2_responses[0]

        expanded_csv = clean_response(expanded_csv_raw, "csv")
        n_new_entries = get_csv_text_n_entries(expanded_csv)
        pbar.update(n_new_entries)

        # Concatenate new entries
        csv_data = csv_data.strip() + '\n' + '\n'.join(expanded_csv.strip().splitlines()[1:])
        csv_data = alphabetize_csv_text(csv_data)
    
    # Check if we hit max iterations
    if i >= max_iters and get_csv_text_n_entries(csv_data) < min_entries:
        logger.warning(f"{step_name.title()} hit max iterations ({max_iters}) with only {get_csv_text_n_entries(csv_data)} entries (target: {min_entries})")

    # Save CSV data
    step_memory_dir = os.path.join(args.memory_dir, step_name)
    os.makedirs(step_memory_dir, exist_ok=True)
    csv_fn = os.path.join(step_memory_dir, f'{step_name}.csv')
    with open(csv_fn, 'w', encoding='utf-8') as f:
        f.write(csv_data)
    
    logger.info(f"{step_name.title()} saved to: {csv_fn}")
    logger.info(f"{step_name.title()} step completed")
    return True


def run_lexicon_step(args, llm_client):
    """Run the lexicon generation step."""
    required_files = {
        'phonology': 'phonology.txt',
        'grammar': 'grammar.txt'
    }
    prompt_files = ['lex_step1_extract.txt', 'lex_step2_expand.txt']
    
    return _run_iterative_csv_step(
        args, llm_client, 'lexicon', required_files, prompt_files,
        'lexicon_min_entries', 'lexicon_n_per_iter', 'lexicon_extra_sleep',
        'lexicon_max_iters'
    )


def run_translation_step(args, llm_client):
    """Run the single-sentence translation step."""
    # Required files
    required_files = {
        'phonology': 'phonology.txt',
        'grammar': 'grammar.txt',
    }
    
    # Optional files
    optional_files = {
        'lexicon': 'lexicon.csv',
    }
    
    # Load files with optional lexicon support
    files = load_required_files(args.memory_dir, required_files)
    if files is None:
        return False
    
    # Try to load optional lexicon
    try:
        lexicon_path = os.path.join(args.memory_dir, 'lexicon', 'lexicon.csv')
        if os.path.exists(lexicon_path):
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                files['lexicon'] = f.read()
    except Exception as e:
        logger.warning(f"Could not load lexicon: {e}")
    
    # Load prompt
    prompt_dir = os.path.join(args.prompt_dir, 'translation')
    raw_prompt = PromptManager.load_prompt(os.path.join(prompt_dir, 'translation_single.txt'))
    
    # Prepare lexicon section based on availability
    if 'lexicon' in files:
        lexicon_section = f"""It has the following lexicon:

=== START ===
{files['lexicon']}
=== END ==="""
    else:
        lexicon_section = """Note: No specific lexicon has been provided. You will need to create appropriate vocabulary words that follow the phonological and morphological patterns of the language."""
    
    # Prepare kwargs for prompt formatting
    kwargs = {
        'phonology': files['phonology'],
        'grammar': files['grammar'],
        'lexicon_section': lexicon_section,
        'input_sentence': args.translation_input_sentence
    }
    
    # Format and generate
    prompt = PromptManager.format_prompt(raw_prompt, **kwargs)
    logger.info(f"Translation prompt: {prompt}")
    
    # Generate translation
    _, content = llm_client.generate_and_extract(prompt, do_sleep=False)
    content = clean_response(content, "json")
    
    # Save translation result
    metadata = {
        'input_sentence': args.translation_input_sentence,
        'lexicon_available': 'lexicon' in files
    }
    
    logger.info(f"Saving translation in memory")
    step_memory_dir = os.path.join(args.memory_dir, 'translation')
    save_memory(content, step_memory_dir, 'translation.json', metadata)
    
    logger.info(f"Translation step completed")
    return True