from atlas.index.api import Indexer

__all__ = ["Indexer"]


'''
API design

idx = Indexer("data.lance")

idx.create_index(column, index_type (FTS, vector), (optionlal, if vecotr)model: hr_model_id, **kwargs) -- This should be enough to create the index 
    by determining the column's data-type and choosing the pre-defined default embedding model for each data-type or users model

idx.list_indexe((optional) column) -- Rich/pretty display the table schema with all existing index tpyes for each feature column, if column is not passed


'''