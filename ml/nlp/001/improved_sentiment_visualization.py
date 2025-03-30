import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

def visualize_sentiment_and_inclination(X_tfidf, y_test, y_pred, clf):
    """
    Enhanced visualization function for sentiment analysis results
    
    Parameters:
    -----------
    X_tfidf : scipy.sparse.csr.csr_matrix
        The TF-IDF vectorized test data
    y_test : array-like
        The true labels
    y_pred : array-like
        The predicted labels
    clf : classifier object
        The trained classifier with predict_proba method
    """
    # Set up the visualization style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#FF5733', '#33A1FF']  # Red for negative, Blue for positive
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Movie Review Sentiment Analysis Visualization', fontsize=20, fontweight='bold')
    
    # 1. Confusion Matrix with Enhanced Styling
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, 
                        index=['Actual Negative', 'Actual Positive'], 
                        columns=['Predicted Negative', 'Predicted Positive'])
    
    # Plot heatmap with annotations
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1, 
                annot_kws={"size": 16, "weight": "bold"})
    
    # Calculate and display accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    ax1.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontsize=14, fontweight='bold')
    
    # 2. Classification Report Visualization
    ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    
    # Get classification report as dictionary
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Extract metrics for visualization
    metrics = ['precision', 'recall', 'f1-score']
    neg_metrics = [report['neg'][metric] for metric in metrics]
    pos_metrics = [report['pos'][metric] for metric in metrics]
    
    # Set up bar positions
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create grouped bar chart
    ax2.bar(x - width/2, neg_metrics, width, label='Negative', color=colors[0], alpha=0.8)
    ax2.bar(x + width/2, pos_metrics, width, label='Positive', color=colors[1], alpha=0.8)
    
    # Add value labels on top of bars
    for i, v in enumerate(neg_metrics):
        ax2.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    for i, v in enumerate(pos_metrics):
        ax2.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Customize the plot
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Performance Metrics by Sentiment Class', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend(loc='lower right')
    
    # 3. Dimensionality Reduction for Visualization
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    
    # Convert sparse matrix to dense for PCA
    X_dense = X_tfidf.toarray()
    
    # Apply PCA first to reduce dimensions to 50
    pca = PCA(n_components=min(50, X_dense.shape[1]))
    X_pca = pca.fit_transform(X_dense)
    
    # Then apply t-SNE for better visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Create a DataFrame for easier plotting
    df_tsne = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'label': y_test
    })
    
    # Plot each class with different colors
    for label, color in zip(['neg', 'pos'], colors):
        indices = df_tsne['label'] == label
        ax3.scatter(
            df_tsne.loc[indices, 'x'], 
            df_tsne.loc[indices, 'y'],
            c=color,
            label='Negative' if label == 'neg' else 'Positive',
            alpha=0.7,
            edgecolors='w',
            s=100
        )
    
    ax3.set_title('t-SNE Visualization of Movie Reviews', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.set_xlabel('t-SNE Feature 1')
    ax3.set_ylabel('t-SNE Feature 2')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Prediction Probability Distribution
    ax4 = plt.subplot2grid((2, 3), (1, 2))
    
    # Get prediction probabilities if the classifier supports it
    try:
        # For classifiers that support predict_proba
        probs = clf.predict_proba(X_tfidf)
        
        # Create histogram of positive class probabilities
        pos_probs = probs[:, 1] if probs.shape[1] == 2 else None
        
        if pos_probs is not None:
            # Create a colormap that transitions from red to blue
            cmap = mcolors.LinearSegmentedColormap.from_list('RedBlue', colors)
            
            # Plot histogram with color gradient
            n, bins, patches = ax4.hist(pos_probs, bins=20, alpha=0.7, edgecolor='black')
            
            # Color the bins according to their position
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers)
            col /= max(col)
            
            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cmap(c))
            
            ax4.set_title('Distribution of Positive Sentiment Probability', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Probability of Positive Sentiment')
            ax4.set_ylabel('Number of Reviews')
            
            # Add a vertical line at 0.5 threshold
            ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
            ax4.text(0.5, ax4.get_ylim()[1]*0.9, 'Decision Threshold', 
                    rotation=90, va='top', ha='right')
    except:
        # For classifiers that don't support predict_proba
        ax4.text(0.5, 0.5, 'Probability distribution not available for this classifier',
                ha='center', va='center', fontsize=12)
        ax4.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Example usage:
# visualize_sentiment_and_inclination(X_test, y_test, y_pred, clf)

def visualize_sentiment_wordcloud(X, y, tfidf_vectorizer):
    """
    Create word clouds for positive and negative sentiments
    
    Parameters:
    -----------
    X : list
        The original text data
    y : array-like
        The sentiment labels
    tfidf_vectorizer : TfidfVectorizer
        The fitted TF-IDF vectorizer
    """
    from wordcloud import WordCloud
    
    # Convert to numpy arrays for easier indexing
    X = np.array(X)
    y = np.array(y)
    
    # Separate positive and negative reviews
    pos_reviews = X[y == 'pos']
    neg_reviews = X[y == 'neg']
    
    # Combine all positive and negative reviews
    pos_text = ' '.join(pos_reviews)
    neg_text = ' '.join(neg_reviews)
    
    # Create figure for word clouds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Generate word clouds
    pos_wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='Blues',
        max_words=100,
        contour_width=3,
        contour_color='steelblue'
    ).generate(pos_text)
    
    neg_wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='Reds',
        max_words=100,
        contour_width=3,
        contour_color='firebrick'
    ).generate(neg_text)
    
    # Display word clouds
    ax1.imshow(pos_wordcloud, interpolation='bilinear')
    ax1.set_title('Positive Reviews Word Cloud', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(neg_wordcloud, interpolation='bilinear')
    ax2.set_title('Negative Reviews Word Cloud', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# visualize_sentiment_wordcloud(X, y, tfidf_vectorizer)

def visualize_feature_importance(clf, tfidf_vectorizer, top_n=20):
    """
    Visualize the most important features (words) for classification
    
    Parameters:
    -----------
    clf : classifier object
        The trained classifier with coef_ attribute (e.g., LogisticRegression)
    tfidf_vectorizer : TfidfVectorizer
        The fitted TF-IDF vectorizer
    top_n : int
        Number of top features to display
    """
    try:
        # Get feature names and coefficients
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Get coefficients from the classifier
        if hasattr(clf, 'coef_'):
            # For linear models like LogisticRegression
            coefficients = clf.coef_[0]
        elif hasattr(clf, 'feature_importances_'):
            # For tree-based models
            coefficients = clf.feature_importances_
        else:
            print("Classifier doesn't provide feature importance information")
            return
        
        # Create a DataFrame with features and their importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefficients
        })
        
        # Sort by absolute importance
        feature_importance['Abs_Importance'] = abs(feature_importance['Importance'])
        feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)
        
        # Get top positive and negative features
        top_positive = feature_importance[feature_importance['Importance'] > 0].head(top_n)
        top_negative = feature_importance[feature_importance['Importance'] < 0].head(top_n)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot top positive features
        ax1.barh(top_positive['Feature'][::-1], top_positive['Importance'][::-1], color='#33A1FF')
        ax1.set_title('Top Positive Sentiment Words', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Importance')
        
        # Plot top negative features
        ax2.barh(top_negative['Feature'][::-1], top_negative['Importance'][::-1], color='#FF5733')
        ax2.set_title('Top Negative Sentiment Words', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing feature importance: {e}")

# Example usage:
# visualize_feature_importance(clf, tfidf_vectorizer)
