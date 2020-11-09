import time
import pickle
from path import Path
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import torch
import torch.nn as nn

from utils import LABEL_NAME, isnotebook, set_seed, format_time

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


set_seed(seed=228)

def model_train(model, train_data_loader, valid_data_loader, test_data_loader,
                logger, optimizer, scheduler, num_epochs, seed, out_dir):
    # move model to gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    num_gpus = torch.cuda.device_count()
    logger.info("Let's use {} GPUs!".format(num_gpus))

    # Set the seed value all over the place to make this reproducible.
#     set_seed(seed=seed)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []
    print_interval = 100

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    batch_size = train_data_loader.batch_size
    num_batch = len(train_data_loader)
    best_f1_score = {
        "weighted": 0,
        "averaged": 0
    }
    best_test_f1_score = 0

    # For each epoch...
    for epoch_i in range(0, num_epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        logger.info("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epochs))
        logger.info('Training...')

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Measure how long the training epoch takes.
        t_train = time.time()

        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_data_loader), desc="Training Iteration", total=num_batch):
            # Progress update every 100 batches.
            if step % print_interval == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t_train)
                avg_train_loss = total_train_loss / print_interval

                # Report progress.
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.3e} | loss {:5.3f} | Elapsed {:s}'.format(
                    epoch_i+1, step, num_batch, scheduler.get_last_lr()[0], avg_train_loss, elapsed)
                )
                total_train_loss = 0
                training_stats.append(
                    {
                        'epoch': epoch_i + 1,
                        'step': step,
                        'train loss': avg_train_loss,
                    }
                )

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains four pytorch tensors:
            #   "input_ids"
            #   "attention_mask"
            #   "token_type_ids"
            #   "binarized_labels"

            b_review_input_ids = batch["review_input_ids"].to(device)
            b_review_attention_mask = batch["review_attention_mask"].to(device)
            b_review_token_type_ids = batch["review_token_type_ids"].to(device)
            b_agent_input_ids = batch["agent_input_ids"].to(device)
            b_agent_attention_mask = batch["agent_attention_mask"].to(device)
            b_agent_token_type_ids = batch["agent_token_type_ids"].to(device)

            b_binarized_label = batch["binarized_label"].to(device)

            model.zero_grad()
            (loss, _) = model(review_input_ids=b_review_input_ids,
                              review_attention_mask=b_review_attention_mask,
                              review_token_type_ids=b_review_token_type_ids,
                              agent_input_ids=b_agent_input_ids,
                              agent_attention_mask=b_agent_attention_mask,
                              agent_token_type_ids=b_agent_token_type_ids,

                              labels=b_binarized_label
                              )

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.

            if num_gpus > 1:
                total_train_loss += loss.mean().item()
                loss.mean().backward()  # use loss.mean().backward() instead of loss.backward() for multiple gpu trainings
            else:
                total_train_loss += loss.item()
                loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            scheduler.step()
        # End of training epoch

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t_train)

        logger.info("")
        logger.info("  Training epoch took: {:s}".format(training_time))

        # evaluate the model after one epoch.

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        logger.info("")
        logger.info("Validating...")

        t_valid = time.time()
        model.eval()
        ave_valid_loss, valid_f1_table, cm_table, f1_score = model_validate(model=model, data_loader=valid_data_loader)
        # Measure how long this epoch took.
        validation_time = format_time(time.time() - t_valid)

        logger.info("")
        logger.info('| loss {:5.3f} | Elapsed {:s}'.format(ave_valid_loss, validation_time))
        logger.info("  \n{:s}".format(valid_f1_table.to_string()))
        logger.info("")
        logger.info("  \n{:s}".format(cm_table.to_string()))

        # need to store the best model
        for key in best_f1_score.keys():
            if best_f1_score[key] < f1_score[key]:
                # remove the old model:
                file_list = [f for f in out_dir.files() if f.name.endswith(".pt") and f.name.startswith(key)]
                for f in file_list:
                    Path.remove(f)
                model_file = out_dir.joinpath('{:s}_epoch_{:02d}-f1_{:.3f}.pt'.format(
                    key, epoch_i + 1, f1_score[key])
                )
                best_f1_score[key] = f1_score[key]
                if num_gpus > 1:
                    torch.save(model.module.state_dict(), model_file)
                else:
                    torch.save(model.state_dict(), model_file)

        # ========================================
        #               Test
        # ========================================
        logger.info("")
        logger.info("Testing...")

        result_df = model_test(model=model, data_loader=test_data_loader)
    
        y_true = np.array(result_df["review_label"], dtype=np.bool) # This part may need double check
        y_pred = result_df["Probability"] > 0.5

        report = classification_report(y_true, y_pred, output_dict=True)
        metrics_df = pd.DataFrame(report).transpose()

        metrics_df = metrics_df.sort_index()

        weighted_f1_score = metrics_df.loc['weighted avg', 'f1-score']
        averaged_f1_score = metrics_df.loc['macro avg', 'f1-score']

        best_test_f1_score = metrics_df.loc['weighted avg', 'f1-score'] \
            if best_test_f1_score < metrics_df.loc['weighted avg', 'f1-score'] else best_test_f1_score

        metrics_df = metrics_df.astype(float).round(3)

        # Calculate confusion matrix
        tn, fp, fn, tp  = confusion_matrix(y_true, y_pred).ravel()
        cm_df = pd.DataFrame(columns = ['Predicted No', 'Predicted Yes'],  
                       index = ['Actual No', 'Actual Yes']) 
        # adding rows to an empty  
        # dataframe at existing index 
        cm_df.loc['Actual No'] = [tn,fp] 
        cm_df.loc['Actual Yes'] = [fn,tp]
        
        logger.info("use model: {} batch / {} step".format(epoch_i + 1, step))
        logger.info("\n" + "=" * 50)
        logger.info("\n" + metrics_df.to_string())
        logger.info("\n" + "=" * 50)
        logger.info("\n" + cm_df.to_string())
        logger.info("best test F1 score: {}".format(best_test_f1_score))
        logger.info("\n" + "=" * 50)
        # Below is to save the result files
        result_filename = "result_df_epoch_" + str(epoch_i + 1) + ".xlsx"
        result_df.to_excel(out_dir.joinpath(result_filename), index=False)

    logger.info("")
    logger.info("Training complete!")
    logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # Save training_stats to csv file
    pd.DataFrame(training_stats).to_csv(out_dir.joinpath("model_train.log"), index=False)
    return model, optimizer, scheduler


def model_validate(model, data_loader):
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    label_prop = data_loader.dataset.dataset.label_prop()

    total_valid_loss = 0

    batch_size = data_loader.batch_size
    num_batch = len(data_loader)

    y_pred, y_true = [], []

    # Evaluate data
    for step, batch in tqdm(enumerate(data_loader), desc="Validation...", total=num_batch):
        b_review_input_ids = batch["review_input_ids"].to(device)
        b_review_attention_mask = batch["review_attention_mask"].to(device)
        b_review_token_type_ids = batch["review_token_type_ids"].to(device)
        b_agent_input_ids = batch["agent_input_ids"].to(device)
        b_agent_attention_mask = batch["agent_attention_mask"].to(device)
        b_agent_token_type_ids = batch["agent_token_type_ids"].to(device)

        b_binarized_label = batch["binarized_label"].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            (loss, logits,) = model(review_input_ids=b_review_input_ids,
                                    review_attention_mask=b_review_attention_mask,
                                    review_token_type_ids=b_review_token_type_ids,
                                    agent_input_ids=b_agent_input_ids,
                                    agent_attention_mask=b_agent_attention_mask,
                                    agent_token_type_ids=b_agent_token_type_ids,

                                    labels=b_binarized_label)

        total_valid_loss += loss.item()
        ### The sigmoid function is used for the two-class logistic regression, 
        ### whereas the softmax function is used for the multiclass logistic regression
        
        # Version 1
        # numpy_probas = logits.detach().cpu().numpy()
        # y_pred.extend(np.argmax(numpy_probas, axis=1).flatten())
        # y_true.extend(b_binarized_label.cpu().numpy())

        # Version 2
        # transfored_logits = F.log_softmax(logits,dim=1)
        # numpy_probas = transfored_logits.detach().cpu().numpy()
        # y_pred.extend(np.argmax(numpy_probas, axis=1).flatten())
        # y_true.extend(b_binarized_label.cpu().numpy())

        # Version 3
        # transfored_logits = torch.sigmoid(logits)
        # numpy_probas = transfored_logits.detach().cpu().numpy()
        # y_pred.extend(np.argmax(numpy_probas, axis=1).flatten())
        # y_true.extend(b_binarized_label.cpu().numpy())

        # New version - for num_label = 1
        transfored_logits = torch.sigmoid(logits)
        numpy_probas = transfored_logits.detach().cpu().numpy()
        y_pred.extend(numpy_probas)
        y_true.extend(b_binarized_label.cpu().numpy())
        
    # End of an epoch of validation

    # put model to train mode again.
    model.train()

    ave_loss = total_valid_loss / (num_batch * batch_size)

    y_pred = np.array(y_pred)
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    
    # Below is in case the input and target are not the same data format
    y_pred = np.array(y_pred, dtype=np.bool)
    y_true = np.array(y_true, dtype=np.bool)
    
    
    # compute the various f1 score for each label
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    # metrics_df = pd.DataFrame(0, index=LABEL_NAME, columns=["Precision", "Recall", "F1","support"])
    # metrics_df.Precision = precision_recall_fscore_support(y_true, y_pred)[0]
    # metrics_df.Recall = precision_recall_fscore_support(y_true, y_pred)[1]
    # metrics_df.F1 = precision_recall_fscore_support(y_true, y_pred)[2]
    # metrics_df.support = precision_recall_fscore_support(y_true, y_pred)[3]

    # y_pred = np.array(y_pred)
    # y_pred[y_pred < 0] = 0
    # y_pred[y_pred > 0] = 1
    # y_pred = np.array(y_pred, dtype=np.bool)
    # y_true = np.array(y_true, dtype=np.bool)

    # metrics_df = pd.DataFrame(0, index=LABEL_NAME, columns=["Precision", "Recall", "F1"], dtype=np.float)
    # # or_y_pred = np.zeros(y_pred.shape[0], dtype=np.bool)
    # # or_y_true = np.zeros(y_true.shape[0], dtype=np.bool)
    # for i in range(len(LABEL_NAME)):
    #     metrics_df.iloc[i] = precision_recall_fscore_support(
    #         y_true=y_true[:, i], y_pred=y_pred[:, i], average='binary', zero_division=0)[0:3]

        # or_y_pred = or_y_pred | y_pred[:, i]
        # or_y_true = or_y_true | y_true[:, i]

    metrics_df = metrics_df.sort_index()
    # metrics_df.loc['Weighted Average'] = metrics_df.transpose().dot(label_prop)
    # metrics_df.loc['Average'] = metrics_df.mean()

    # metrics_df.loc['Weighted Average', 'F1'] = 2 / (1/metrics_df.loc['Weighted Average', "Recall"] +
    #                                                 1/metrics_df.loc['Weighted Average', "Precision"])
    # metrics_df.loc['Average', 'F1'] = 2 / (1/metrics_df.loc['Average', "Recall"] +
    #                                        1/metrics_df.loc['Average', "Precision"])

    weighted_f1_score = metrics_df.loc['weighted avg', 'f1-score']
    averaged_f1_score = metrics_df.loc['macro avg', 'f1-score']

    # Calculate confusion matrix
    tn, fp, fn, tp  = confusion_matrix(y_true, y_pred).ravel()
    cm_df = pd.DataFrame(columns = ['Predicted No', 'Predicted Yes'],  
                   index = ['Actual No', 'Actual Yes']) 
    # adding rows to an empty  
    # dataframe at existing index 
    cm_df.loc['Actual No'] = [tn,fp] 
    cm_df.loc['Actual Yes'] = [fn,tp]

    # pooled_f1_score = f1_score(y_pred=or_y_pred, y_true=or_y_true)

    return ave_loss, metrics_df, cm_df,{
        "weighted": weighted_f1_score,
        "averaged": averaged_f1_score,
    }


def model_test(model, data_loader):
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    num_batch = len(data_loader)
    # Below need to modify if change the input
    review_id, review_label, hmd_text, head_cust_text = [], [], [], []
    agent = []
    pred_logits = []

    # Evaluate data
    for step, batch in tqdm(enumerate(data_loader), desc="Inference...", total=num_batch):
        if "anecdote_lead_final" in batch.keys():
            review_label.extend(batch["anecdote_lead_final"])
        review_id.extend(batch["_id"].tolist())
        hmd_text.extend(batch["hmd_comments"])
        head_cust_text.extend(batch["head_cust"])
        agent.extend(batch["new_transcript_agent"])

        b_review_input_ids = batch["review_input_ids"].to(device)
        b_review_attention_mask = batch["review_attention_mask"].to(device)
        b_review_token_type_ids = batch["review_token_type_ids"].to(device)
        b_agent_input_ids = batch["agent_input_ids"].to(device)
        b_agent_attention_mask = batch["agent_attention_mask"].to(device)
        b_agent_token_type_ids = batch["agent_token_type_ids"].to(device)


        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            (logits,) = model(review_input_ids=b_review_input_ids,
                              review_token_type_ids=b_review_token_type_ids,
                              review_attention_mask=b_review_attention_mask,
                              agent_input_ids=b_agent_input_ids,
                              agent_token_type_ids=b_agent_token_type_ids,
                              agent_attention_mask=b_agent_attention_mask
                              )

        if logits.detach().cpu().numpy().size == 1:
            pred_logits.extend(logits.detach().cpu().numpy().reshape(1,))  
        else:
            pred_logits.extend(logits.detach().cpu().numpy())
            
    # End of an epoch of validation
    # put model to train mode again.
    model.train()
    pred_logits = np.array(pred_logits)
    pred_prob = np.exp(pred_logits)
    pred_prob = pred_prob / (1 + pred_prob)
    pred_label = pred_prob.copy()
    pred_label[pred_label < 0.5] = 0
    pred_label[pred_label >= 0.5] = 1
    # compute the f1 score for each tags
    d = {'Probability':pred_prob,'Anecdotes Prediction':pred_label}
    pred_df = pd.DataFrame(d, columns=['Probability','Anecdotes Prediction'])
    result_df = pd.DataFrame(
        {
            "review_id": review_id,
            "hmd_text": hmd_text,
            "head_cust_text": head_cust_text,
            "agent": agent
        }
    )
    if len(review_label) != 0:
        result_df["review_label"] =  [x.item() for x in review_label] 
    return pd.concat([result_df, pred_df], axis=1).set_index("review_id")
