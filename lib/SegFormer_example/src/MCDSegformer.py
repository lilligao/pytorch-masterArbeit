class MCDSegformer(L.LightningModule):
    def __init__(self):
        super().__init__()

        if config.WEIGHTS == 'imagenet':
            L.seed_everything(42)       # reproducibility
        elif config.WEIGHTS == 'random':
            L.seed_everything(time.time(), workers=True)    # random seed for random weights in the decoder 

        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)
        self.model = SegformerForSemanticSegmentation(model_config)
        self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False,
            hidden_dropout_prob=config.DROPOUT_RATE,
            attention_probs_dropout_prob=config.DROPOUT_RATE,
            classifier_dropout_prob=config.DROPOUT_RATE,
            drop_path_rate=config.DROPOUT_RATE,
        )   

        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, eps=1e-7)

        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.val_ece = torchmetrics.CalibrationError(task='multiclass', n_bins=10, num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)


    def training_step(self, batch, batch_index):
        images, _, labels = batch

        loss, logits = self.model(images, labels.squeeze(dim=1))
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_index):
        images, _, labels = batch
        
        ### Auskommentierte Sachen sind für MC-Dropout, das sollte man dann aber nicht während dem Training durchlaufen lassen, sondern im Anschluss, wenn man einen finalen Checkpoint hat
        ## Activate dropout layers
        # for m in self.model.modules():
        #     if m.__class__.__name__.startswith('Dropout'):
        #         m.train()

        ## For 5 samples
        # sample_outputs = torch.empty(size=[config.NUM_SAMPLES, images.shape[0], config.NUM_CLASSES, images.shape[-2], images.shape[-1]], device=self.device)
        # for i in range(config.NUM_SAMPLES):
        #     loss, logits = self.model(images, labels.squeeze(dim=1))
        #     upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

        #     sample_outputs[i] = torch.softmax(upsampled_logits, dim=1)
        
        # probability_map = torch.mean(sample_outputs, dim=0)
        # prediction_map = torch.argmax(probability_map, dim=1, keepdim=True)
        # standard_deviation_map = torch.std(sample_outputs, dim=0)
        # entropy_map = torch.sum(-probability_map * torch.log(probability_map + 1e-6), dim=1, keepdim=True)

        ## Beispiel für die Berechnung der Uncertainty Metrics mit der entropy_map. Analog könnte man es natürlich auch mit der standard_deviation_map machen.
        # p_accurate_certain, p_inaccurate_uncertain, pavpu = compute_uncertainty_metrics(images, labels.squeeze(dim=1), prediction_map, entropy_map)
        # self.log('pAccCer_entropy', p_accurate_certain, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log('pInaUnc_entropy', p_inaccurate_uncertain, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log('pavpu_entropy', pavpu, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        loss, logits = self.model(images, labels.squeeze(dim=1))
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

        probability_map = torch.softmax(upsampled_logits, dim=1)

        self.val_iou(probability_map, labels.squeeze(dim=1))
        self.val_ece(probability_map, labels.squeeze(dim=1))
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_ece', self.val_ece, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    

def compute_uncertainty_metrics(images, labels, prediction, uncertainty):
    """
    Computes uncertainty metrics for a given set of images, labels, predictions, and uncertainties. Meant to be used for semantic segmentation.

    Args:
        images (torch.Tensor): A tensor of shape (N, C, H, W) representing the input images.
        labels (torch.Tensor): A tensor of shape (N, H, W) representing the ground-truth labels.
        prediction (torch.Tensor): A tensor of shape (N, H, W) representing the predicted labels.
        uncertainty (torch.Tensor): A tensor of shape (N, H, W) representing the uncertainty scores (predictive_uncertainty, entropy, standard deviation,...).

    Returns:
        Tuple of three floats representing the following metrics:
        - p_accurate_certain: The proportion of pixels that are accurate and certain.
        - p_uncertain_inaccurate: The proportion of pixels that are inaccurate and uncertain.
        - pavpu: The proportion of accurate pixels among the uncertain ones.
    """
    uncertainty_threshold = torch.mean(uncertainty) # Hier kannst du prinizipiell natürlich auch andere Thresholds dir überlegen. Ich habe hier einfach den Durchschnitt genommen. 

    binary_accuracy_map = (prediction == labels).float()

    # count the number of pixels that are accurate and certain
    n_ac = torch.sum((binary_accuracy_map == 1) & (uncertainty < uncertainty_threshold))

    # count the number of pixels that are accurate and uncertain
    n_au = torch.sum((binary_accuracy_map == 1) & (uncertainty >= uncertainty_threshold))

    # count the number of pixels that are inaccurate and certain
    n_ic = torch.sum((binary_accuracy_map == 0) & (uncertainty < uncertainty_threshold))

    # count the number of pixels that are inaccurate and uncertain
    n_iu = torch.sum((binary_accuracy_map == 0) & (uncertainty >= uncertainty_threshold))

    # compute the metrics
    p_accurate_certain = n_ac / (n_ac + n_ic)
    p_uncertain_inaccurate = n_iu / (n_ic + n_iu)
    pavpu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

    return p_accurate_certain, p_uncertain_inaccurate, pavpu