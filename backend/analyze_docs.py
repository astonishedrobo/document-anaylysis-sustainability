import argparse
import dotenv
import os
from utils.analysis.analyzer import analyze_doc_rag
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the documentation of a project.')
    parser.add_argument('--doc_path', type=str, help='The path to the project to analyze.')
    parser.add_argument('--prompt_path', type=str, help='The path to the prompt file (txt).')
    parser.add_argument('--augment_links', type=bool)
    parser.add_argument('--country', type=str, help='The country of the project.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the .env file
    dotenv_path = os.path.join(os.getcwd(), '.env')
    dotenv.load_dotenv(dotenv_path)

    # Load the prompt
    with open(args.prompt_path, 'r') as file:
        prompt = json.load(file)

    # Analyze the documentation
    doc_analysis = {}
    for i, question in enumerate(prompt["questions"].keys()):
        previous_context = None
        query = prompt["questions"][question] + '\n' + prompt["postfix"][question]
        print(query)
        if i > 0:
            # Previous context is the previous answers (j<i) concatenated
            previous_context = '\n\n'.join([str(doc_analysis[f"Q{j+1}"]) for j in range(i)])
            # previous_context = ""
            # previous_context = doc_analysis[f"Q{i}"]
        doc_analysis[f"Q{i+1}"] = analyze_doc_rag(args.doc_path, question=query, augment_link=args.augment_links, previous_context=previous_context, model_name='gpt-3.5-turbo')
    print(doc_analysis)
    # Save the analysis in Json file
    with open(f'doc_analysis_{args.country}.json', 'w') as file:
        json.dump(doc_analysis, file, indent=4)

if __name__ == "__main__":
    main()
