import clip
import torch
import numpy as np


# TODO: move this to a module where it makes sense
def global_min(similarity_dict):
    similarities_all = torch.cat(list(similarity_dict.values()), dim=0)
    return torch.min(similarities_all, dim=0)[0]

class CLIPLoss:

    def __init__(self, clip_models, captions, representation, input_size, device, clip_bs, rep_bs, aggregate=global_min,
                 smaller_is_better=True):
        # here we need a build clip function
        self.clip_models = clip_models
        self.representation = representation
        self.device = device
        self.clip_bs = clip_bs
        self.rep_bs = rep_bs
        self.input_size = input_size
        # determine latent space properties by sending random image to decoder

        random_test_img = torch.rand((1, 3, input_size, input_size)).to(self.device)
        self.latent_shape = self.representation.from_image(random_test_img).cpu().numpy().shape
        self.captions = captions
        self.text_features = self.process_captions(self.captions)
        self.aggregate = aggregate
        self.smaller_is_better = smaller_is_better

    def process_captions(self, captions):
        tokens = clip.tokenize(captions).to(self.device)
        clip_text_features = {}
        for clip_string in self.clip_models.keys():
            clip_model = self.clip_models[clip_string][0]
            with torch.no_grad():
                text_features = clip_model.encode_text(tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
            clip_text_features[clip_string] = text_features
        return clip_text_features

    def candidates_to_image_tensor(self, candidate_solutions):
        candidate_solutions = np.array(candidate_solutions)
        candidate_solutions = np.reshape(candidate_solutions,
                                         (candidate_solutions.shape[0],) + self.latent_shape[1:]).astype('float32')
        candidate_solutions = torch.from_numpy(candidate_solutions).to(self.device)

        return self.latent_to_image_tensor(candidate_solutions)

    def latent_to_image_tensor(self, latent_vectors):
        batches = torch.split(latent_vectors, self.rep_bs)
        images = []
        # TODO: actually, this batching should probably be taken care of in representation,
        # that's what we originally wrote the class for
        for batch in batches:
            image_batch = self.representation.to_image(batch)
            images.extend(image_batch)
        return images

    def score_images(self, images):

        similarity_dict = {}
        for clip_string in self.clip_models.keys():
            clip_model, preprocess = self.clip_models[clip_string]
            #TODO: find a elegant way to use no_grad for cma-es and with grad for the gradient
            #with torch.no_grad():
            images_pp = preprocess(images)
            image_features = clip_model.encode_image(images_pp)
            image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
            similarity_dict[clip_string] = torch.mm(self.text_features[clip_string], image_features_norm.T)
        return similarity_dict

    def loss(self, candidate_solutions):
        # TODO: maybe a class CMAConverter makes sense here that handles the conversion from image to candidate and vice versa?
        images = torch.cat([image[None] for image in self.candidates_to_image_tensor(candidate_solutions)])
        similarities = self.loss_grad(images)
        return similarities.cpu().detach().numpy()

    def loss_grad(self,images):
        similarities = self.aggregate(self.score_images(images))
        if self.smaller_is_better:
            similarities = -similarities
        return similarities

class ShiftedCLIPLoss(CLIPLoss):
    def __init__(self, clip_strings, captions, representation, input_size, device, clip_bs, rep_bs, gap_vectors,
                 aggregate=global_min, smaller_is_better=True):
        self.gap_vectors = dict(
            [(clip_string, gap_vector.float()) for (clip_string, gap_vector) in gap_vectors.items()])
        super().__init__(clip_strings, captions, representation, input_size, device, clip_bs, rep_bs, aggregate, smaller_is_better)

    def process_captions(self, captions):
        tokens = clip.tokenize(captions).to(self.device)
        clip_text_features = {}
        for clip_string in self.clip_models.keys():
            clip_model = self.clip_models[clip_string][0]
            gap_vector = self.gap_vectors[clip_string]
            with torch.no_grad():
                text_features = clip_model.encode_text(tokens).float()
                text_features += gap_vector
                text_features /= text_features.norm(dim=-1, keepdim=True)
            clip_text_features[clip_string] = text_features
        return clip_text_features

    def score_images(self, images):
        similarity_dict = {}
        for clip_string in self.clip_models.keys():
            clip_model, preprocess = self.clip_models[clip_string]
            gap_vector = self.gap_vectors[clip_string]
            #with torch.no_grad():
            images_pp = preprocess(images)
            image_features = clip_model.encode_image(images_pp).float()
            image_features -= gap_vector
            image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
            similarity_dict[clip_string] = torch.mm(self.text_features[clip_string], image_features_norm.T)
        return similarity_dict
