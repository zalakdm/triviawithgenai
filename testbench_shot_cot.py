import csv
from genai_agent_simple import OpenaiAgent

if __name__=="__main__":
    # Parse the CSV file
    with open("testbench.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        headers = next(reader)
        data = list(reader)

    # Get the correct answers
    correct_answers = []

    # Instantiate a HIP agent
    agent = OpenaiAgent()

    # Get the user's responses
    user_responses = []
    for i, row in enumerate(data):
        print(f"Processing question {i+1}/{len(data)}...")
        answer_choices = [row[headers.index("answer_0")],
                        row[headers.index("answer_1")],
                        row[headers.index("answer_2")],
                        row[headers.index("answer_3")]]
        correct_answers.append(answer_choices.index(row[headers.index("correct")]))
        response = agent.get_response(row[headers.index("question")], answer_choices)
        user_responses.append(response)

    # Calculate the score
    score = 0
    answers = []
    for i in range(len(data)):
        if user_responses[i] == correct_answers[i]:
            score += 1
            answers += [[1, user_responses[i], correct_answers[i]]]
        else:
            answers += [[0, user_responses[i], correct_answers[i]]]

    # Display the score
    print(f"\nScore: {score}/{len(data)}")
    print(f"Accuracy: {score/len(data)*100:.1f}%")
    print(f"\nDetailed results:")
    for i, result in enumerate(answers):
        status = "✓" if result[0] == 1 else "✗"
        print(f"Q{i+1}: {status} (AI: {result[1]}, Correct: {result[2]})") 