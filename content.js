function getTweetText() {
  const tweetNode = document.querySelector('[data-testid="tweetText"]');
  const authorNode = document.querySelector('[data-testid="User-Name"]');
  
  if (tweetNode && authorNode) {
    const tweetText = tweetNode.innerText;
    const authorInfo = {
      name: authorNode.innerText,
      verified: document.querySelector('[data-testid="icon-verified"]') !== null,
      handle: document.querySelector('[data-testid="User-Name"] span').innerText
    };
    
    console.log("Extracted Tweet:", tweetText);
    console.log("Author Info:", authorInfo);
    
    checkForMisinformation(tweetText, authorInfo);
  }
}

function checkForMisinformation(text, authorInfo) {
  const credibleSources = [
    {handle: "@US_FDA", name: "U.S. FDA"},
    {handle: "@CDCgov", name: "CDC"},
    {handle: "@WHO", name: "WHO"},
    {handle: "@NIH", name: "NIH"},
    {handle: "@HHSGov", name: "HHS.gov"}
  ];
  
  const isCredibleSource = credibleSources.some(source => 
    authorInfo.handle.includes(source.handle) || 
    authorInfo.name.includes(source.name)
  );
  
  if (isCredibleSource && authorInfo.verified) {
    displayCredibleSourceBadge(text, authorInfo);
  } else {
    sendToClassificationAPI(text, authorInfo);
  }
}

function displayCredibleSourceBadge(text, authorInfo) {
  const tweetNode = document.querySelector('[data-testid="tweetText"]');
  if (!tweetNode) return;
  
  const badge = document.createElement('div');
  badge.className = 'misinfo-badge';
  badge.innerHTML = `
    <div class="misinfo-header" style="background-color: #4caf50;">
      <span class="misinfo-title">Credible Source</span>
      <span class="misinfo-confidence">Official Information</span>
    </div>
    <div class="misinfo-details">
      <p><strong>Source Verification:</strong> This content is from ${authorInfo.name}, a verified official health authority.</p>
      <p class="misinfo-disclaimer">Information from official health authorities is generally reliable, but always verify with multiple sources.</p>
    </div>
  `;
  
  tweetNode.parentNode.insertBefore(badge, tweetNode.nextSibling);
}

function sendToClassificationAPI(text, authorInfo) {
  fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      text: text
    }),
  })
  .then(response => response.json())
  .then(result => {
    const features = {
      word_count: text.split(/\s+/).filter(word => word.length > 0).length,
      hashtag_count: (text.match(/#\w+/g) || []).length,
      mention_count: (text.match(/@\w+/g) || []).length,
      url_count: (text.match(/https?:\/\/[^\s]+/g) || []).length
    };
    
    result.features = features;
    
    const opinionResult = detectOpinion(text);
    result.isOpinion = opinionResult.isOpinion;
    result.opinionWords = opinionResult.opinionWords;
    
    displayClassificationResult(text, result);
  })
  .catch(error => {
    console.error('Error:', error);
  });
}

function detectOpinion(text) {
  const opinionIndicators = [
    'i think', 'i believe', 'i feel', 'in my opinion', 'i reckon',
    'i guess', 'probably', 'possibly', 'maybe', 'perhaps',
    'seems like', 'appears to be', 'might be', 'could be',
    'should', 'would', 'must be', 'personally', 'from my perspective',
    'as far as i can tell', 'i suspect', 'i doubt', 'i wonder',
    'i hope', 'i wish', 'i expect', 'i assume', 'i suppose',
    'apparently', 'allegedly', 'supposedly', 'arguably',
    'best', 'worst', 'good', 'bad', 'terrible', 'great', 'awful',
    'amazing', 'wonderful', 'horrible', 'fantastic', 'disappointing'
  ];
  
  const firstPersonPronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'];
  
  const lowerText = text.toLowerCase();
  
  const foundIndicators = opinionIndicators.filter(indicator => 
    lowerText.includes(indicator)
  );
  
  const foundPronouns = firstPersonPronouns.filter(pronoun => {
    const regex = new RegExp(`\\b${pronoun}\\b`, 'i');
    return regex.test(lowerText);
  });
  
  const allOpinionWords = [...foundIndicators, ...foundPronouns];
  
  const isOpinion = 
    foundIndicators.length >= 2 || 
    (foundPronouns.length > 0 && foundIndicators.length > 0);
  
  return {
    isOpinion: isOpinion,
    opinionWords: allOpinionWords
  };
}

function displayClassificationResult(text, result) {
  const tweetNode = document.querySelector('[data-testid="tweetText"]');
  if (!tweetNode) return;
  
  let contentType = 'Factual Claim';
  let contentTypeClass = '';
  
  if (result.isOpinion) {
    contentType = 'Opinion';
    contentTypeClass = 'opinion-content';
  }
  
  chrome.storage.local.get(['confidenceThreshold'], function(data) {
    const threshold = data.confidenceThreshold || 0.67;
    
    if (result.confidence < threshold) {
      result.prediction = 'uncertain';
    }
    
    if (result.isOpinion && result.prediction === 'fake') {
      result.prediction = 'opinion';
    }
    
    let badgeColor, badgeText;
    if (result.prediction === 'fake' || result.prediction === 1) {
      badgeColor = '#ff4d4d';
      badgeText = 'Potential Misinformation';
    } else if (result.prediction === 'opinion') {
      badgeColor = '#9370db';
      badgeText = 'Opinion Detected';
    } else if (result.prediction === 'uncertain') {
      badgeColor = '#ffcc00';
      badgeText = 'Uncertain';
    } else {
      badgeColor = '#4caf50';
      badgeText = 'Likely Reliable';
    }
    
    const confidencePercent = Math.round(result.confidence * 100);
    
    const badge = document.createElement('div');
    badge.className = 'misinfo-badge';
    badge.innerHTML = `
      <div class="misinfo-header" style="background-color: ${badgeColor};">
        <span class="misinfo-title">${badgeText}</span>
        <span class="misinfo-confidence">${confidencePercent}% confidence</span>
      </div>
      <div class="misinfo-details">
        <p><strong>Content Type:</strong> ${contentType}</p>
        ${result.isOpinion ? `<p><strong>Opinion Indicators:</strong> ${result.opinionWords.join(', ')}</p>` : ''}
        <p><strong>Tweet Analysis:</strong></p>
        <ul>
          <li>Words: ${result.features.word_count}</li>
          <li>Hashtags: ${result.features.hashtag_count}</li>
          <li>Mentions: ${result.features.mention_count}</li>
          <li>URLs: ${result.features.url_count}</li>
        </ul>
        <p class="misinfo-explanation"><strong>Explanation:</strong> ${result.explanation ? result.explanation.join(', ') : 'No explanation provided'}</p>
        <p class="misinfo-disclaimer">This is an automated assessment. Please verify information from reliable sources.</p>
      </div>
    `;
    
    tweetNode.parentNode.insertBefore(badge, tweetNode.nextSibling);
  });
}

document.addEventListener('click', () => {
  setTimeout(getTweetText, 500);
});


window.addEventListener('load', () => {
  setTimeout(getTweetText, 1000);
});


setInterval(getTweetText, 5000);


function setupHoverAnalysis() {
    
    const tweetSelector = 'article[data-testid="tweet"]';
    
    
    document.addEventListener('mouseover', function(event) {
        const tweetElement = event.target.closest(tweetSelector);
        if (!tweetElement) return;
        
        
        if (tweetElement.dataset.hoverAnalyzed === 'true') return;
        
        
        tweetElement.dataset.hoverAnalyzed = 'true';
        
        
        const tweetTextElement = tweetElement.querySelector('[data-testid="tweetText"]');
        if (!tweetTextElement) return;
        
        const tweetText = tweetTextElement.textContent;
        
        
        analyzeTweetOnHover(tweetText, tweetElement);
    });
}


async function analyzeTweetOnHover(tweetText, tweetElement) {
    try {
        const response = await fetch('http://localhost:5000/analyze_opinion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: tweetText,
                check_misinfo: true
            })
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const result = await response.json();
        
        
        displayHoverResults(result, tweetElement);
        
        
        chrome.storage.local.get(['tweetsAnalyzed', 'misinfoDetected', 'opinionsDetected'], function(data) {
            const tweetsAnalyzed = (data.tweetsAnalyzed || 0) + 1;
            const misinfoDetected = (data.misinfoDetected || 0) + (result.misinfo_prediction === 'fake' ? 1 : 0);
            const opinionsDetected = (data.opinionsDetected || 0) + (result.is_opinion ? 1 : 0);
            
            chrome.storage.local.set({
                tweetsAnalyzed: tweetsAnalyzed,
                misinfoDetected: misinfoDetected,
                opinionsDetected: opinionsDetected
            });
        });
        
    } catch (error) {
        console.error('Error analyzing tweet on hover:', error);
    }
}

function displayHoverResults(result, tweetElement) {
    
    let resultsContainer = tweetElement.querySelector('.tweet-analysis-results');
    if (!resultsContainer) {
        resultsContainer = document.createElement('div');
        resultsContainer.className = 'tweet-analysis-results';
        tweetElement.appendChild(resultsContainer);
    }
    
    
    resultsContainer.innerHTML = '';
    
    
    const opinionSection = document.createElement('div');
    opinionSection.className = `opinion-result ${result.is_opinion ? 'opinion' : 'fact'}`;
    
    let opinionDetails = `
        <div class="result-header">
            <span class="result-icon">${result.is_opinion ? '💭' : '📝'}</span>
            <span class="result-title">${result.is_opinion ? 'Opinion Detected' : 'Likely Factual'}</span>
        </div>
        <div class="result-details">
            <span>Sentiment: ${result.sentiment} (${(result.sentiment_scores.compound * 100).toFixed(0)}%)</span>
    `;
    
    
    if (result.opinion_indicators && result.opinion_indicators.length > 0) {
        opinionDetails += `
            <div class="opinion-indicators">
                <span>Indicators:</span>
                <ul>
                    ${result.opinion_indicators.map(indicator => `<li>${indicator}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    opinionDetails += `</div>`;
    opinionSection.innerHTML = opinionDetails;
    
    
    chrome.storage.local.get(['confidenceThreshold'], function(data) {
        const threshold = data.confidenceThreshold || 0.67; 
        
        
        let misinfoClass = result.misinfo_prediction === 'fake' ? 'fake' : 'real';
        let misinfoTitle = result.misinfo_prediction === 'fake' ? 'Potential Misinformation' : 'Likely Reliable';
        let misinfoIcon = result.misinfo_prediction === 'fake' ? '⚠️' : '✓';
        
        
        if (result.misinfo_confidence < threshold) {
            misinfoClass = 'uncertain';
            misinfoTitle = 'Uncertain';
            misinfoIcon = '❓';
        }
        
        
        const misinfoSection = document.createElement('div');
        misinfoSection.className = `misinfo-result ${misinfoClass}`;
        misinfoSection.innerHTML = `
            <div class="result-header">
                <span class="result-icon">${misinfoIcon}</span>
                <span class="result-title">${misinfoTitle}</span>
            </div>
            <div class="result-details">
                <span>Confidence: ${(result.misinfo_confidence * 100).toFixed(0)}%</span>
                <span class="explanation">${result.misinfo_explanation.join(', ')}</span>
            </div>
        `;
        
        
        resultsContainer.appendChild(opinionSection);
        resultsContainer.appendChild(misinfoSection);
        
        
        addHoverResultsStyles();
    });
}


function addHoverResultsStyles() {
    
    if (document.getElementById('hover-analysis-styles')) return;
    
    const styleElement = document.createElement('style');
    styleElement.id = 'hover-analysis-styles';
    styleElement.textContent = `
        .tweet-analysis-results {
            margin-top: 10px;
            padding: 10px;
            border-radius: 8px;
            background-color: #f8f9fa;
            border: 1px solid #e1e8ed;
        }
        
        .opinion-result, .misinfo-result {
            margin-bottom: 8px;
            padding: 8px;
            border-radius: 6px;
        }
        
        .opinion-result.opinion {
            background-color: #e8f4fd;
            border-left: 4px solid #1da1f2;
        }
        
        .opinion-result.fact {
            background-color: #eaf7ee;
            border-left: 4px solid #17bf63;
        }
        
        .misinfo-result.fake {
            background-color: #feeaea;
            border-left: 4px solid #e0245e;
        }
        
        .misinfo-result.real {
            background-color: #eaf7ee;
            border-left: 4px solid #17bf63;
        }
        
        .misinfo-result.uncertain {
            background-color: #fff9e6;
            border-left: 4px solid #ffcc00;
        }
        
        .result-header {
            display: flex;
            align-items: center;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .result-icon {
            margin-right: 8px;
            font-size: 16px;
        }
        
        .result-details {
            font-size: 13px;
            color: #657786;
        }
        
        .explanation {
            display: block;
            margin-top: 5px;
            font-style: italic;
        }
    `;
    
    document.head.appendChild(styleElement);
}


document.addEventListener('DOMContentLoaded', function() {
    setupHoverAnalysis();
    
    
    const observer = new MutationObserver(function(mutations) {
        setupHoverAnalysis();
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
});
