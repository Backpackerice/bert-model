import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, EmbeddingBag, CrossEntropyLoss, MultiLabelSoftMarginLoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, AdamW

from utils import LABEL_NAME


class ReviewClassification(BertPreTrainedModel):
    def __init__(self, config,
                 add_agent_text, agent_text_heads):
        """
        :param config: Bert configuration, can set up some parameters, like  output_attention, output_hidden_states
        :param add_agent_text: whether to use the non text feature, and how.
                It can have three options: None, "concat" and "attention"
        :param agent_text_heads: number of the heads in agent attention mechanism. Only useful if add_agent_text are set to
                "attention"
        """
        super().__init__(config)
        # self.num_labels = 2
        self.add_agent_text = add_agent_text

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        embedding_size = config.hidden_size

        if self.add_agent_text == "concat":
            embedding_size = 2 * embedding_size
        elif self.add_agent_text == "attention":
            self.agent_attention = nn.MultiheadAttention(embedding_size, num_heads=agent_text_heads)
        else:
            # don't use the information in Agent text
            pass

        # if self.add_non_text == "concat":
        #     size_out = 32
        #     self.non_text_dense = nn.Linear(len(GL_CATEGORY) + 1, size_out)  # 1 refer to overall rating
        #     self.non_text_bn = nn.BatchNorm1d(size_out)
        #     embedding_size = embedding_size + size_out
        # elif self.add_non_text == "attention":
        #     size_out = config.hidden_size
        #     self.non_text_embedding = nn.Embedding(len(GL_CATEGORY), size_out)
        #     self.non_text_attention = nn.MultiheadAttention(size_out, num_heads=non_text_heads)
        # else:
        #     # don't use the information in non text features(GL group and overall rating)
        #     pass

        self.classifier = nn.Linear(embedding_size, 1) # self.classifier = nn.Linear(embedding_size, len(LABEL_NAME)) # bias: If set to False, the layer will not learn an additive bias
        self.init_weights()

        print(
            """            
            add agent text         :{}
            agent text multi-head  :{}
            """.format(self.add_agent_text, agent_text_heads)
        )

    def forward(
            self,
            review_input_ids=None,
            review_attention_mask=None,
            review_token_type_ids=None,
            agent_input_ids=None,
            agent_attention_mask=None,
            agent_token_type_ids=None,
            labels=None,
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        review_outputs = self.bert(
            review_input_ids,
            attention_mask=review_attention_mask,
            token_type_ids=review_token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )
        if self.add_agent_text is not None:
            # means that self.add_agent_text is "concat" or "attention"
            # TODO: we can try that agent_outputs do not share the same parameter
            agent_outputs = self.bert(
                agent_input_ids,
                attention_mask=agent_attention_mask,
                token_type_ids=agent_token_type_ids,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
            )
        ## If need to change which hidden states to embed, refer to before output.
        # the shape of the outputs can be found in here:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        # outputs[0] is the last_hidden_state with the shape (batch_size, sequence_length, hidden_size)
        # outputs[1] is the pooler_output with the shape (batch_size, hidden_size)
        # outputs[2] is the tuple of hidden_states, with the shape
        # (1 + #layers,  (batch_size, sequence_length, hidden_size)), 1 refers to the input embedding
        # outputs[3] is the tuple of attentions, each with the shape
        # (batch_size, num_heads, sequence_length, sequence_length)
        # outputs[2] and outputs[3] are available only output_hidden_states=True and output_attentions=True

        if self.add_agent_text == "attention":
            # want to take it as key and value, we need to transpose its shape according to the document
            # https://pytorch.org/docs/master/generated/torch.nn.MultiheadAttention.html
            review_hidden_states = review_outputs[0].transpose(0, 1)  # before trans: (bs, seq_len, hidden_size)

            # want to take it as query, we need the it has the shape (#target_seq_len, batch_size, embedding_size)
            agent_hidden_states = agent_outputs[0].mean(axis=1).unsqueeze(dim=0)  # (1, batch_size, hidden_size)

            attn_output, _ = self.agent_attention(agent_hidden_states, review_hidden_states, review_hidden_states)
            feature = attn_output.squeeze()  # (batch_size, seq_len)
        else:
            # don't use the attention mechanism
            # have two options in here to make classification:
            # 1. only use the first CLS token to make classification
            feature = review_outputs[1]  # (batch_size, seq_len) -? Should it be (batch_size, hidden_size)
            # 2. use mean of the hidden state
            #feature = review_outputs[0].mean(axis=1)

        if self.add_agent_text == "concat":
            feature = torch.cat([feature, agent_outputs[1]], axis=1)
       

        # nn.CrossEntropyLoss applies F.log_softmax and nn.NLLLoss internally on your input,
        # so you should pass the raw logits to it.

        # torch.nn.functional.binary_cross_entropy takes logistic sigmoid values as inputs
        # torch.nn.functional.binary_cross_entropy_with_logits takes logits as inputs
        # torch.nn.functional.cross_entropy takes logits as inputs (performs log_softmax internally)
        # torch.nn.functional.nll_loss is like cross_entropy but takes log-probabilities (log-softmax) values as inputs

        # CrossEntropyLoss takes prediction logits (size: (N,D)) and target labels (size: (N,)) 
        # CrossEntropyLoss expects logits i.e whereas BCELoss expects probability value
        logits = self.classifier(feature).squeeze()

        outputs = (logits,)  # + outputs[2:]  # add hidden states and attention if they are here


        if labels is not None:
            ##### original
            # loss_fct = MultiLabelSoftMarginLoss()
            # loss = loss_fct(logits, labels)
            # outputs = (loss,) + outputs
            #### Version 1 try
            # pos_weight = dataset.label_proportion.iloc[0]/dataset.label_proportion.iloc[1]

            # Version 1.1 for weight
            # weight = torch.tensor([0.101521, 0.898479]) # hard code from entire training dataset
            # pos_weight = weight[labels.data.view(-1).long()].view_as(labels)
            # Version 1.2 for weight
            #pos_weight=torch.tensor(2.0)
            # Version 1.3 for weight
            #weight = torch.tensor([1.0, 8.85]) # hard code from entire training dataset
            #pos_weight = weight[labels.data.view(-1).long()].view_as(labels)
            
            loss_fct = nn.BCEWithLogitsLoss().cuda() #pos_weight=pos_weight
            loss = loss_fct(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs
            ### Version 2 try
            # loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # outputs = (loss,) + outputs

        return outputs  # (loss, logits, hidden_states, attentions)



