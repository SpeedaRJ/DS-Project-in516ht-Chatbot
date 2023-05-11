from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, PreProcessor, PDFToTextConverter, TransformersReader, Seq2SeqGenerator
from haystack.nodes.answer_generator.transformers import _BartEli5Converter
from haystack.pipelines import Pipeline

import pathlib as pl


def build_document_store():
    converter = PDFToTextConverter(remove_numeric_tables=True)
    extracted = converter.convert(file_path=pl.Path(
        "../data/raw/sustainability-report-2020.pdf"), meta=False, encoding="UTF-8")[0]
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="sentence",
        split_length=4,
        split_respect_sentence_boundary=False,
        split_overlap=0
    )
    cleaned = preprocessor.process([extracted])
    document_store = FAISSDocumentStore(
        faiss_index_factory_str='Flat', similarity="dot_product")
    document_store.write_documents(cleaned)
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=True,
    )
    document_store.update_embeddings(retriever)
    document_store.save("document_store.faiss")


def read_document_store():
    return FAISSDocumentStore.load("document_store.faiss")


def build_retriever(ds):
    retriever = DensePassageRetriever(
        document_store=ds,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=True
    )
    return retriever


def build_reader():
    return TransformersReader(model_name_or_path="..\initial_moddeling\distilbert-qa\distilbert-nlb-qa", use_gpu=True)


def build_generator():
    return Seq2SeqGenerator(model_name_or_path="../initial_moddeling/t5-qa/t5-small-finetuned-squadv2-finetuned-NLB-QA/", input_converter=_BartEli5Converter())


def build_pipeline(model, retriever):
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipe.add_node(component=model, name="Model", inputs=["Retriever"])
    return pipe


def run_query(model, query):
    return model.run(query=query, params={"Model": {"top_k": 3}})
