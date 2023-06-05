from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, PreProcessor, PDFToTextConverter, TransformersReader, Seq2SeqGenerator
from haystack.nodes.answer_generator.transformers import _BartEli5Converter
from haystack.pipelines import Pipeline


def read_document_store():
    return FAISSDocumentStore.load(index_path="document_store.faiss", config_path="document_store.json")


def build_retriever(ds):
    retriever = DensePassageRetriever(
        document_store=ds,
        query_embedding_model="SpeedaRJ/dpr-fb-finetuned-nlb-question-encoder",
        passage_embedding_model="SpeedaRJ/dpr-fb-finetuned-nlb-context-encoder",
        use_gpu=True
    )
    return retriever


def build_reader():
    return TransformersReader("SpeedaRJ/roberta-base-nlb-finetuned", use_gpu=True)


def build_generator():
    return Seq2SeqGenerator("SpeedaRJ/t5-base-nlb-finetuned", input_converter=_BartEli5Converter(), use_gpu=True)


def build_pipeline(model, retriever):
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipe.add_node(component=model, name="Model", inputs=["Retriever"])
    return pipe


def run_query(model, query):
    return model.run(query=query, params={"Model": {"top_k": 1}})
