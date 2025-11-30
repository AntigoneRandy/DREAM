import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch.nn.functional as F


class BLIP2Classifier(torch.nn.Module):
    def __init__(
        self,
        device,
        file_path,
        model_name="blip2_image_text_matching",
        model_type="pretrain",
        batch_size=32,
    ):
        super(BLIP2Classifier, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            model_name, model_type, device=device, is_eval=False
        )
        self.model.eval()
        print(f"BLIP2: Loading prompt tuning weights from {file_path}")
        soft_prompts = torch.load(file_path, map_location=device)
        self.soft_prompts = torch.nn.parameter.Parameter(soft_prompts).to(self.device)



    def _load_images(self, inputs):
        """Internal method for unified image data loading"""
        loaded_images = []
        for item in inputs:
            if isinstance(item, str):
                # Handle path input
                img = Image.open(item).convert("RGB")
            elif isinstance(item, Image.Image):
                # Handle direct image object input
                img = item.convert("RGB") if item.mode != "RGB" else item
            else:
                raise TypeError(f"Unsupported input type: {type(item)}")
            loaded_images.append(img)
        return loaded_images
    
    def forward(self, inputs):
        if isinstance(inputs, (str, Image.Image)):
            inputs = [inputs]
        with torch.no_grad():
            # Load and preprocess images
            images = self._load_images(inputs)
            processed_images = [self.vis_processors["eval"](img).to(self.device) for img in images]

            batched_outputs = []
            total = len(processed_images)
            for start in range(0, total, self.batch_size):
                end = min(start + self.batch_size, total)
                image = torch.stack(processed_images[start:end], dim=0).to(self.device)

                N = image.shape[0]

                with self.model.maybe_autocast():
                    image_embeds = self.model.ln_vision(self.model.visual_encoder(image))
                image_embeds = image_embeds.float()

                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

                query_output = self.model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                image_feats = F.normalize(
                    self.model.vision_proj(query_output.last_hidden_state), dim=-1
                )
                text_feats = F.normalize(
                    self.soft_prompts, dim=-1
                )
                sims = torch.matmul(image_feats, text_feats.T) / 0.01
                sim, _ = torch.max(sims, dim=1)

                probs = torch.softmax(sim, dim=-1)
                batched_outputs.append(probs)

        return torch.cat(batched_outputs, dim=0) if batched_outputs else torch.tensor([])
