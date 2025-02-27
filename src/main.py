from transformers import pipeline


qa_pipeline = pipeline(
    "question-answering",
    model="MilyaShams/rubert-russian-qa-sberquad"
)


def answer_question(context, question):
    result = qa_pipeline(question=question, context=context)
    return result['answer']


print(answer_question(
    'Машинное обучение — это область искусственного интеллекта, которая '
    'изучает методы построения алгоритмов, способных обучаться на данных. '
    'Оно широко применяется в анализе данных, компьютерном зрении и '
    'обработке естественного языка.',
    'Где применяется машинное обучение?'
))
