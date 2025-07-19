import openai
import re

with open("openaikey.txt") as f:
    openai.api_key = f.read().strip()

class OpenaiAgent:
    def __init__(self):
        # Define few-shot examples based on the biology questions in the dataset
        self.few_shot_examples = [
            {
                "question": "GMOs are created by ________",
                "choices": [
                    "generating genomic DNA fragments with restriction endonucleases",
                    "introducing recombinant DNA into an organism by any means", 
                    "overexpressing proteins in E. coli",
                    "all of the above"
                ],
                "correct_answer": "introducing recombinant DNA into an organism by any means",
                "reasoning": "Let me think about this step by step:\n1. GMOs (Genetically Modified Organisms) are created by introducing foreign DNA into an organism\n2. This involves recombinant DNA technology\n3. The key is that any method of introducing recombinant DNA creates a GMO\n4. The other options are specific techniques, but the broad definition is 'introducing recombinant DNA by any means'\n5. Therefore, the answer is 'introducing recombinant DNA into an organism by any means'"
            },
            {
                "question": "What is a biomarker?",
                "choices": [
                    "the color coding of different genes",
                    "a protein that is uniquely produced in a diseased state",
                    "a molecule in the genome or proteome", 
                    "a marker that is genetically inherited"
                ],
                "correct_answer": "a protein that is uniquely produced in a diseased state",
                "reasoning": "Let me think about this step by step:\n1. A biomarker is a measurable indicator of a biological state or condition\n2. It's specifically used to detect diseases or medical conditions\n3. The key characteristic is that it's 'uniquely produced' in a diseased state\n4. While biomarkers can be molecules in genome/proteome, the most specific definition is a protein uniquely produced in disease\n5. Therefore, the answer is 'a protein that is uniquely produced in a diseased state'"
            },
            {
                "question": "Which scientific concept did Charles Darwin and Alfred Wallace independently discover?",
                "choices": [
                    "mutation",
                    "natural selection",
                    "overbreeding", 
                    "sexual reproduction"
                ],
                "correct_answer": "natural selection",
                "reasoning": "Let me think about this step by step:\n1. Charles Darwin and Alfred Wallace are both credited with the theory of evolution\n2. They independently arrived at the same conclusion about how species evolve\n3. The key concept they discovered was the mechanism of evolution\n4. This mechanism is called 'natural selection' - the process where organisms better adapted to their environment survive and reproduce\n5. Therefore, the answer is 'natural selection'"
            }
        ]
    
    def _create_chain_of_thought_prompt(self, question, answer_choices):
        """
        Creates a prompt with chain of thought reasoning using few-shot examples.
        """
        # Start with system instruction
        system_prompt = """You are an expert biology tutor. Your task is to answer multiple choice questions by thinking through the problem step-by-step and then selecting the most accurate answer.

Important instructions:
1. Read the question carefully and understand what is being asked
2. Think through the problem step-by-step, showing your reasoning
3. Consider each answer choice and eliminate obviously incorrect ones
4. Choose the answer that best matches the question
5. End your response with 'ANSWER: [exact text of chosen answer]'
6. Make sure your final answer matches one of the given choices exactly"""

        # Build the few-shot examples with reasoning
        examples = ""
        for i, example in enumerate(self.few_shot_examples, 1):
            choices_text = "\n".join([f"{j+1}. {choice}" for j, choice in enumerate(example['choices'])])
            examples += f"""
Example {i}:
Question: {example['question']}
Choices:
{choices_text}

{example['reasoning']}

ANSWER: {example['correct_answer']}

"""

        # Format the current question
        current_choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(answer_choices)])
        
        # Combine everything into the final prompt
        user_prompt = f"""{examples}
Now answer this question:
Question: {question}
Choices:
{current_choices_text}

Think through this step by step:"""

        return system_prompt, user_prompt

    def _extract_answer_from_response(self, response_text, answer_choices):
        """
        Extracts the final answer from a chain of thought response.
        Looks for 'ANSWER:' pattern and matches it to the choices.
        """
        # Look for the ANSWER: pattern
        answer_pattern = r'ANSWER:\s*(.+)'
        match = re.search(answer_pattern, response_text, re.IGNORECASE)
        
        if match:
            extracted_answer = match.group(1).strip()
            # Try to match the extracted answer to the choices
            for i, answer_choice in enumerate(answer_choices):
                if extracted_answer.lower() == answer_choice.lower():
                    return i
        
        # If no ANSWER: pattern found, try to match the last sentence or phrase
        # Split by lines and look for the last non-empty line
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            for i, answer_choice in enumerate(answer_choices):
                if last_line.lower() == answer_choice.lower():
                    return i
        
        # Fallback: try partial matching on the entire response
        for i, answer_choice in enumerate(answer_choices):
            if (answer_choice.lower() in response_text.lower() or 
                response_text.lower() in answer_choice.lower()):
                return i
        
        return -1

    def get_response(self, question, answer_choices):
        """
        Calls the OpenAI 3.5 API to generate a response using chain of thought reasoning.
        The response is then parsed to extract the final answer and matched to one of the answer choices.
        If the response does not match any answer choice, -1 is returned.

        Args:
            question: The question to be asked.
            answer_choices: A list of answer choices.

        Returns:
            The index of the answer choice that matches the response, or -1 if the response
            does not match any answer choice.
        """

        # Create the prompt with chain of thought reasoning
        system_prompt, user_prompt = self._create_chain_of_thought_prompt(question, answer_choices)

        # Call the OpenAI 3.5 API with chain of thought prompting
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Slightly higher temperature for more creative reasoning
            max_tokens=300,   # Allow more tokens for reasoning
        )
        
        # Extract response text
        response_text = response['choices'][0]['message']['content'].strip()  # type: ignore

        # Extract the final answer from the chain of thought response
        return self._extract_answer_from_response(response_text, answer_choices) 