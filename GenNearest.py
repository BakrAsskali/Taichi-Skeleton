import numpy as np
import cv2
import os
import pickle
import sys
import math
from sklearn.neighbors import NearestNeighbors


class GenNearest:
    """
    A class that generates images based on skeleton poses using nearest neighbor matching.
    For each input skeleton, it finds the most similar skeleton in the training data
    and returns the corresponding image.
    """

    def __init__(self, videoSkeTgt, use_reduced=False):
        """
        Initialize the generator with video skeleton data.

        Parameters:
        -----------
        videoSkeTgt : VideoSkeleton object
            Contains paired skeleton-image data from source video
        use_reduced : bool, optional (default=False)
            If True: uses reduced skeleton (13 joints × 2 coords = 26 features)
            If False: uses full skeleton (33 joints × 3 coords = 99 features)
        """
        # Store reference to input video and configuration
        self.videoSkeletonTarget = videoSkeTgt
        self.use_reduced = use_reduced
        self.cached_images = []  # Will store normalized images from video
        print("Caching images and building nearest neighbor index...")

        # Initialize list to store skeleton features for nearest neighbor search
        self.skeleton_features = []

        # Process each frame in the video skeleton data
        for i in range(len(self.videoSkeletonTarget.ske)):
            # Load and normalize image to [0,1] range
            img = self.videoSkeletonTarget.readImage(i)
            img = img.astype(np.float32) / 255.0  # Convert to float and normalize
            self.cached_images.append(img)

            # Extract skeleton features based on configuration
            if use_reduced:
                # Get reduced skeleton representation (26 features)
                ske_features = self.videoSkeletonTarget.ske[i].__array__(reduced=True).flatten()
            else:
                # Get full skeleton representation (99 features)
                ske_features = self.videoSkeletonTarget.ske[i].__array__(reduced=False).flatten()
            self.skeleton_features.append(ske_features)

        # Convert list of features to numpy array for efficient processing
        self.skeleton_features = np.array(self.skeleton_features)

        # Calculate feature dimension for debugging/verification
        feature_dim = self.skeleton_features.shape[1] if len(self.skeleton_features.shape) > 1 else len(
            self.skeleton_features[0])
        print(f"Feature dimension: {feature_dim}")

        # Initialize the nearest neighbor search model
        self.nn = NearestNeighbors(
            n_neighbors=1,  # Only find single closest match
            algorithm='ball_tree',  # Use ball tree algorithm (efficient for medium dimensions)
            metric='euclidean',  # Use Euclidean distance for similarity
            n_jobs=-1  # Use all CPU cores for parallel processing
        )
        # Train the nearest neighbor model with our skeleton features
        #  (really just organizing the data for efficient search)

        self.nn.fit(self.skeleton_features)

        print(f"Cached {len(self.cached_images)} images and built nearest neighbor index")

    def generate(self, ske):
        """
        Generate an image for a given skeleton pose.

        Parameters:
        -----------
        ske : Skeleton object
            The input skeleton pose to find a match for

        Returns:
        --------
        numpy.ndarray
            Either a red error image (if ske is None) or
            the image corresponding to the most similar skeleton pose
        """
        # Handle case where no skeleton is provided
        if ske is None:
            # Create red error image with same dimensions as cached images
            red_image = np.zeros_like(self.cached_images[0])
            red_image[:, :] = [0, 0, 1]  # BGR format: pure red
            return red_image

        # Convert input skeleton to feature vector
        if self.use_reduced:
            query_features = ske.__array__(reduced=True).flatten()
        else:
            query_features = ske.__array__(reduced=False).flatten()

        # Reshape to 2D array as required by sklearn (samples × features)
        query_features = query_features.reshape(1, -1)

        # Find most similar skeleton in our database
        distances, indices = self.nn.kneighbors(query_features)
        closest_idx = indices[0][0]  # Get index of best match

        # Return the corresponding cached image
        return self.cached_images[closest_idx]