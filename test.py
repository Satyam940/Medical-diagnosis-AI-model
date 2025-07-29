from g4f.client import Client
client = Client()

a = input("Enter the : ")
response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages =  [  
            {'role':'system','content': 'You are the professional doctor. Patients will come and tell their symptoms in the local language. You have to convert it back to medical symptom names — for example, "ulti" to "vomiting" — and return it in the form of a string.'},
            {'role':'user','content': a},  
            # {'role':'assistant','content': 'you should chat in 2 or 3 word in chat to act like a actual person.'},
            ]

        )

print(response.choices[0].message.content)