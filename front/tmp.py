from src.generate_embegging import AnswersClustering

text = "Какой-то текст"
AC = AnswersClustering()
AC.add_answer(text)
answer, cluster = AC.get_answers(), AC.get_clusters()


