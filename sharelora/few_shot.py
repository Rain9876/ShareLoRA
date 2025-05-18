
MATH_4cot_json =[
    {
      "problem": "Problem:\nFind the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
      "solution": "Solution:\nThe expressions inside each square root must be non-negative.\nTherefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$.\nAlso, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.\nTherefore, the domain of the expression is $\\boxed{[2,5)}$.",
      "Answer": "[2,5)"
    },
    {
      "problem": "Problem:\nIf $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
      "solution": "Solution:\nWe have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$",
      "Answer": "24"
    },
    {
      "problem": "Problem:\nTerrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
      "solution": "Solution:\nIf Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$: \\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}",
      "Answer": "16"
    },
    {
      "problem": "Problem:\nIf the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
      "solution": "Solution:\nIf we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$",
      "Answer": "-\\frac{2}{3}"
    }
  ]



GSM8K_8cot_json = [
  {
    "problem": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "solution": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6"
  },
  {
    "problem": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "solution": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5"
  },
  {
    "problem": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
    "solution": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39"
  },
  {
    "problem": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
    "solution": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8"
  },
  {
    "problem": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
    "solution": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9"
  },
  {
    "problem": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
    "solution": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29"
  },
  {
    "problem": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
    "solution": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33"
  },
  {
    "problem": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
    "solution": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8"
  }
]


def apply_chat_template(thoughts):
    chats = []
    for entry in thoughts:
        user_message = {"role": "user", "content": entry["problem"]}
        assistant_message = {"role": "assistant", "content": entry["solution"]}
        chats.append(user_message)
        chats.append(assistant_message)
    return chats


def chat_template(example_text):
    """
    General chat template function that accepts a raw chain-of-thought example string
    and returns a list of two messages (user and assistant). It detects whether the
    example uses "Q:"/ "A:" or "Problem:"/ "Solution:" markers.
    """
    question = None
    chain_of_thought = None

    # Check for "Q:" marker first.
    if "Q:" in example_text:
        q_match = re.search(r"Q:\s*(.*?)\n", example_text, re.DOTALL)
        a_match = re.search(r"A:\s*(.*)", example_text, re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
        if a_match:
            chain_of_thought = a_match.group(1).strip()
    # Otherwise, check for "Problem:" marker.
    elif "Problem:" in example_text:
        q_match = re.search(r"Problem:\s*(.*?)(?:\nSolution:)", example_text, re.DOTALL)
        a_match = re.search(r"Solution:\s*(.*)", example_text, re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
        if a_match:
            chain_of_thought = a_match.group(1).strip()

    # Fallback if markers not found.
    if not question:
        question = "No question found"
    if not chain_of_thought:
        chain_of_thought = example_text

    # Try to detect if a Final Answer line is already present.
    final_answer_pattern = r'Final Answer:\s*The final answer is\s*\$(.*?)\$'
    final_ans_match = re.search(final_answer_pattern, chain_of_thought)
    if final_ans_match:
        final_ans = final_ans_match.group(1).strip()
    else:
        # Otherwise, try extracting a number (integer or float) from the solution.
        num_match = re.search(r'(-?\d+(\.\d+)?)', chain_of_thought)
        final_ans = num_match.group(1) if num_match else "N/A"
        # Append the Final Answer line if missing.
        chain_of_thought += f"\nFinal Answer: The final answer is ${final_ans}$. I hope it is correct."

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": chain_of_thought}
    ]
    return messages
