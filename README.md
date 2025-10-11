# ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline

**Project Page:** [http://conlangcrafter.github.io](http://conlangcrafter.github.io)  
**Paper:** [ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline](https://arxiv.org/abs/2508.06094)

We introduce a fully automated system for constructing languages (conlangs) using large language models. Our multi-stage pipeline creates coherent, diverse artificial languages with their own phonology, grammar, lexicon, and translation capabilities.

## Code

### Supported Models

- **Google Gemini**: gemini-2.5-pro, gemini-1.5-flash
- **DeepSeek**: DeepSeek-R1 (via Together API)

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Generate a language:**
   ```bash
   python src/run_pipeline.py --model gemini-2.5-pro
   ```

### Directory Structure

```
src/                    # Core source code
├── run_pipeline.py     # Main pipeline script
├── llm_client.py       # LLM API clients
├── pipeline_steps.py   # Language generation steps
└── utils.py           # Utility functions

prompts/               # Prompt templates
├── phonology/         # Phonology generation prompts
├── grammar/           # Grammar generation prompts
├── lexicon/           # Lexicon building prompts
└── translation/       # Translation prompts

output/                # Generated languages (created automatically)
```

### Configuration

The system supports various parameters for customizing language generation:

```bash
python src/run_pipeline.py \
    --model gemini-2.5-pro \
    --steps phonology,grammar,lexicon,translation \
    --custom-constraints "Use only 3 vowels" \
    --translation-sentence "Hello, world!"
```

### API Keys

You'll need API keys for the language models:

- **Google Gemini**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **DeepSeek**: Get from [Together AI](https://api.together.xyz/settings/api-keys)

Add these to your `.env` file (copy from `.env.example`).

## Citation

If you use ConlangCrafter in your research, please cite:

```bibtex
@article{conlangcrafter2025,
    title={ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline},
    author={Morris Alper and Moran Yanuka and Raja Giryes and Ga{\v{s}}per Begu{\v{s}}},
    year={2025},
    eprint={2508.06094},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2508.06094}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.