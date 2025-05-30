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
    displayCredibleSourceBadge(text, authorInfo, tweetElement);
  } else {
    sendToClassificationAPI(text, authorInfo);
  }
}

function displayCredibleSourceBadge(text, authorInfo, tweetElement) {
  if (!tweetElement) {
    console.error("No tweet element provided to displayClassificationResult")
    return
  }

  const loadingIndicator = tweetElement.querySelector(".loading-indicator")
  if (loadingIndicator) {
    loadingIndicator.remove()
  }

  const existingBadges = document.querySelectorAll(".misinfo-badge")
  existingBadges.forEach((badge) => badge.remove())

  const badge = document.createElement("div")
  badge.className = "misinfo-badge"

  badge.style.padding = "0"
  badge.style.marginTop = "10px"
  badge.style.borderRadius = "8px"
  badge.style.fontFamily = "Arial, sans-serif"
  badge.style.fontSize = "14px"
  badge.style.lineHeight = "1.4"
  badge.style.boxShadow = "0 2px 5px rgba(0,0,0,0.1)"
  badge.style.overflow = "hidden"
  badge.style.backgroundColor = "white"

  
  const header = document.createElement("div")
  header.style.padding = "10px 14px"
  header.style.display = "flex"
  header.style.justifyContent = "space-between"
  header.style.alignItems = "center"
  header.style.color = "white"
  header.style.fontWeight = "600"

  
  if (result.prediction === "non_covid") {
    header.style.backgroundColor = "#1da1f2" 
    header.innerHTML = `<strong>ℹ️ NOT COVID RELATED</strong>` 
  }
  
  else if (result.prediction === "fake") {
    header.style.backgroundColor = "#ff6b6b"
    header.innerHTML = `<strong>⚠️ Potential Misinformation</strong>` 
  } else if (result.isOpinion) {
    header.style.backgroundColor = "#ffd166"
    header.style.color = "#333" 
    header.innerHTML = `<strong>💭 Opinion Content</strong>` 
  } else {
    header.style.backgroundColor = "#06d6a0"
    header.innerHTML = `<strong>✓ Likely Factual</strong>` 
  }

  badge.appendChild(header)

  
  const details = document.createElement("div")
  details.style.padding = "12px 14px"
  details.style.backgroundColor = "white"
  details.style.color = "#333"

  
  if (result.prediction === "non_covid") {
    details.innerHTML += `<div><strong>Content Type:</strong> Non-COVID Content</div>`
  } else {
    details.innerHTML += `<div><strong>Content Type:</strong> ${result.isOpinion ? "Opinion" : "Factual Claim"}</div>`
  }


  details.innerHTML += `<div style="font-style: italic; margin-top: 8px; font-size: 12px; padding: 5px; background-color: rgba(0,0,0,0.05); border-radius: 4px;">
    This is an automated assessment. Please verify information from reliable sources.
  </div>`

  badge.appendChild(details)

  
  const tweetTextElement = tweetElement.querySelector('[data-testid="tweetText"]')
  if (tweetTextElement) {
    tweetTextElement.parentNode.insertBefore(badge, tweetTextElement.nextSibling)
  }
}

function sendToClassificationAPI(text, authorInfo) {
  
  const isCovidRelated = detectCovidContent(text);
  
  if (!isCovidRelated) {
    
    const nonCovidResult = {
      prediction: 'non_covid',
      confidence: 0.95,
      explanation: ["This tweet does not appear to contain COVID-related content or misinformation."],
      isOpinion: detectOpinion(text).isOpinion,
      opinionWords: detectOpinion(text).opinionWords,
      features: {
        hashtag_count: (text.match(/#\w+/g) || []).length,
        mention_count: (text.match(/@\w+/g) || []).length,
        url_count: (text.match(/https?:\/\/[^\s]+/g) || []).length
      },
      is_covid_related: false
    };
    
    
    const tweetElement = document.querySelector('[data-testid="tweetText"]').closest('article');
    
    
    displayClassificationResult(text, nonCovidResult, tweetElement);
    
    
    updateAnalysisStats(nonCovidResult);
    return;
  }
  
  
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
      hashtag_count: (text.match(/#\w+/g) || []).length,
      mention_count: (text.match(/@\w+/g) || []).length,
      url_count: (text.match(/https?:\/\/[^\s]+/g) || []).length
    };
    
    result.features = features;
    result.is_covid_related = true;
    
    const opinionResult = detectOpinion(text);
    result.isOpinion = opinionResult.isOpinion;
    result.opinionWords = opinionResult.opinionWords;
    
    
    const tweetElement = document.querySelector('[data-testid="tweetText"]').closest('article');
    
    displayClassificationResult(text, result, tweetElement);
    updateAnalysisStats(result);
  })
  .catch(error => {
    console.error('Error:', error);
    
    
    const errorResult = {
      prediction: 'unknown',
      confidence: 0.5,
      explanation: ["Unable to analyze this content due to a technical error."],
      isOpinion: false,
      features: {},
      is_covid_related: detectCovidContent(text)
    };
    
    
    const tweetElement = document.querySelector('[data-testid="tweetText"]').closest('article');
    
    displayClassificationResult(text, errorResult, tweetElement);
  });
}


function detectCovidContent(text) {
  const covidKeywords = [
    'covid', 'coronavirus', 'sars-cov-2', 'pandemic', 'lockdown',
    'quarantine', 'social distancing', 'mask mandate', 'vaccine', 'vaccination',
    'pfizer', 'moderna', 'astrazeneca', 'johnson & johnson', 'j&j',
    'booster', 'delta variant', 'omicron', 'alpha variant', 'beta variant',
    'herd immunity', 'antibodies', 'pcr test', 'rapid test', 'antigen test',
    'hydroxychloroquine', 'remdesivir', 'ventilator', 'icu', 'respirator',
    'n95', 'ppe', 'frontline workers', 'healthcare workers', 'wfh',
    'work from home', 'remote work', 'zoom', 'virtual meeting', 'cdc',
    'who', 'world health organization', 'fauci', 'surgeon general'
  ];
  
  const lowerText = text.toLowerCase();
  
  
  const matchedKeywords = covidKeywords.filter(keyword => lowerText.includes(keyword));
  

  if (text.length < 100) {
    
    return matchedKeywords.length >= 2 || 
           lowerText.includes('covid') || 
           lowerText.includes('coronavirus') ||
           lowerText.includes('sars-cov-2');
  } else {
    
    if (matchedKeywords.length > 0) {
      
      
      if (matchedKeywords.includes('vaccine') || matchedKeywords.includes('vaccination')) {
        
        return lowerText.includes('covid') || 
               lowerText.includes('coronavirus') || 
               lowerText.includes('pandemic');
      }
      return true;
    }
    return false;
  }
}


function covidKeywordsFound(text) {
  const covidKeywords = [
    'covid', 'coronavirus', 'sars-cov-2', 'pandemic', 'lockdown',
    'vaccine', 'vaccination', 'pfizer', 'moderna', 'astrazeneca',
    'booster', 'delta', 'omicron', 'variant', 'immunity',
    'antibodies', 'pcr test', 'rapid test', 'mask', 'social distancing'
  ];
  
  return covidKeywords.some(keyword => text.includes(keyword));
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
    'amazing', 'wonderful', 'horrible', 'fantastic', 'disappointing',
    'imo', 'imho', 'tbh', 'fwiw', 'just saying', 'my take', 'my view',
    'my thoughts', 'my perspective', 'in my view', 'to me', 'for me'
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
  
  
  const opinionScore = (foundIndicators.length * 0.6) + (foundPronouns.length * 0.3);
  
  const isOpinion = 
    opinionScore >= 1.2 || 
    foundIndicators.length >= 2 || 
    (foundPronouns.length > 1 && foundIndicators.length > 0) ||
    text.toLowerCase().includes('opinion') ||
    text.toLowerCase().includes('i think') ||
    text.toLowerCase().includes('i believe');
  
  return {
    isOpinion: isOpinion,
    opinionWords: allOpinionWords,
    opinionScore: opinionScore
  };
}

function displayClassificationResult(text, result, tweetElement) {
  if (!tweetElement) {
    console.error("No tweet element provided to displayClassificationResult");
    return;
  }
  
  const loadingIndicator = tweetElement.querySelector('.loading-indicator');
  if (loadingIndicator) {
    loadingIndicator.remove();
  }
  
  const existingBadges = document.querySelectorAll('.misinfo-badge');
  existingBadges.forEach(badge => badge.remove());
  
  const badge = document.createElement('div');
  badge.className = 'misinfo-badge';
  
  badge.style.padding = '0';
  badge.style.marginTop = '10px';
  badge.style.borderRadius = '8px';
  badge.style.fontFamily = 'Arial, sans-serif';
  badge.style.fontSize = '14px';
  badge.style.lineHeight = '1.4';
  badge.style.boxShadow = '0 2px 5px rgba(0,0,0,0.1)';
  badge.style.overflow = 'hidden';
  badge.style.backgroundColor = 'white'; 
  
  console.log("Result for confidence calculation:", result);
  
  const confidencePercentage = Math.round((result.confidence || 0) * 100);
  console.log("Calculated confidence percentage:", confidencePercentage);
  
  
  const header = document.createElement('div');
  header.style.padding = '10px 14px';
  header.style.display = 'flex';
  header.style.justifyContent = 'space-between';
  header.style.alignItems = 'center';
  header.style.color = 'white';
  header.style.fontWeight = '600';
  
  
  if (result.prediction === 'non_covid') {
    header.style.backgroundColor = '#1da1f2';
    header.innerHTML = `<strong>ℹ️ NOT COVID RELATED</strong> <span style="float:right">${confidencePercentage}% confidence</span>`;
  } 
  
  else if (result.prediction === 'fake') {
    header.style.backgroundColor = '#ff6b6b';
    header.innerHTML = `<strong>⚠️ Potential Misinformation</strong> <span style="float:right">${confidencePercentage}% confidence</span>`;
  } else if (result.isOpinion) {
    header.style.backgroundColor = '#ffd166';
    header.style.color = '#333'; 
    header.innerHTML = `<strong>💭 Opinion Content</strong> <span style="float:right">${confidencePercentage}% confidence</span>`;
  } else {
    header.style.backgroundColor = '#06d6a0';
    header.innerHTML = `<strong>✓ Likely Factual</strong> <span style="float:right">${confidencePercentage}% confidence</span>`;
  }
  
  badge.appendChild(header);
  
  
  const details = document.createElement('div');
  details.style.padding = '12px 14px';
  details.style.backgroundColor = 'white';
  details.style.color = '#333';
  
  
  if (result.prediction === 'non_covid') {
    details.innerHTML += `<div><strong>Content Type:</strong> Non-COVID Content</div>`;
  } else {
    let contentType = result.isOpinion ? 'Opinion' : 'Factual Claim';
    let factualStatus = result.prediction === 'fake' ? 'Potential Misinformation' : 'Likely Factual';
    details.innerHTML += `<div><strong>Content Type:</strong> ${contentType} (${factualStatus})</div>`;
  }
  
  
  details.innerHTML += `<div><strong>Confidence Score:</strong> ${confidencePercentage}%</div>`;
  details.innerHTML += `<div class="confidence-bar-container" style="background-color: #f0f0f0; height: 8px; border-radius: 4px; margin: 5px 0 10px 0; overflow: hidden;">
    <div style="height: 100%; width: ${confidencePercentage}%; background-color: ${getConfidenceColor(confidencePercentage)};"></div>
  </div>`;
  
  
  details.innerHTML += `<div><strong>Decision Keywords:</strong></div>`;
  details.innerHTML += `<ul style="margin: 5px 0; padding-left: 20px;">`;
  
  
  const lowerText = text.toLowerCase();
  
  
  const isNonCovidContent = result.prediction === 'non_covid' || 
                           (!result.is_covid_related && !covidKeywordsFound(lowerText));
  
  if (isNonCovidContent) {

    if (result.prediction !== 'non_covid') {
      header.style.backgroundColor = '#1da1f2'; 
      header.innerHTML = `<strong>ℹ️ NOT COVID RELATED</strong> <span style="float:right">${confidencePercentage}% confidence</span>`;
    }
    
    
    details.innerHTML += `<li>No COVID-related terms detected</li>`;
    
    
    const generalTopics = extractGeneralTopics(text);
    if (generalTopics.length > 0) {
      generalTopics.slice(0, 2).forEach(topic => {
        details.innerHTML += `<li>Content appears to be about: ${topic}</li>`;
      });
    }
  } else if (result.isOpinion) {
    
    const opinionKeywords = result.opinionWords || [];
    if (opinionKeywords.length > 0) {
      opinionKeywords.slice(0, 3).forEach(word => {
        details.innerHTML += `<li>Opinion indicator: "${word}"</li>`;
      });
    } else {
      details.innerHTML += `<li>Opinion indicators detected</li>`;
    }
  } else {
    
    const covidKeywords = [
      'covid', 'coronavirus', 'sars-cov-2', 'pandemic', 'lockdown',
      'vaccine', 'vaccination', 'pfizer', 'moderna', 'astrazeneca',
      'booster', 'delta', 'omicron', 'variant', 'immunity',
      'antibodies', 'pcr test', 'rapid test', 'mask', 'social distancing'
    ];
    
    const foundKeywords = covidKeywords.filter(keyword => lowerText.includes(keyword));
    
    if (foundKeywords.length > 0) {
      foundKeywords.slice(0, 3).forEach(keyword => {
        details.innerHTML += `<li>COVID term: "${keyword}"</li>`;
      });
    } else if (result.is_covid_related) {
      
      details.innerHTML += `<li>COVID-related content detected</li>`;
    }
    
    
    if (result.prediction === 'fake') {
      const misinfoIndicators = extractMisinfoIndicators(text);
      if (misinfoIndicators.length > 0) {
        misinfoIndicators.slice(0, 2).forEach(indicator => {
          details.innerHTML += `<li>Potential issue: ${indicator}</li>`;
        });
      }
    }
  }
  
  details.innerHTML += `</ul>`;
  
  
  if (result.explanation && result.explanation.length > 0) {
    details.innerHTML += `<div><strong>Explanation:</strong></div>`;
    
    
    const cleanExplanations = result.explanation
      .map(exp => {
        if (typeof exp !== 'string') return null;
        
        
        let cleanExp = exp
          .replace(/Here is the JSON object with my analysis:?\s*```\s*\{[\s\S]*/, '')
          .replace(/\{[\s\S]*\}/, '')
          .replace(/"explanation":\s*"([^"]+)"/, '$1')
          .replace(/"misinformation":\s*\{[^\}]+\}/, '')
          .replace(/\\n/g, ' ')
          .trim();
          
        
        cleanExp = cleanExp.replace(/^"|"$/g, '').trim();
        
        // Skip if empty or just contains JSON syntax
        if (!cleanExp || cleanExp.includes('misinformation:') || 
            cleanExp.startsWith('{') || cleanExp.startsWith('[') ||
            cleanExp.includes('JSON') || cleanExp.includes('```')) {
          return null;
        }
        

        if (cleanExp.length > 0 && cleanExp[0].toLowerCase() === cleanExp[0]) {
          cleanExp = cleanExp.charAt(0).toUpperCase() + cleanExp.slice(1);
        }
        
        
        if (cleanExp.length > 0 && !cleanExp.endsWith('.') && 
            !cleanExp.endsWith('!') && !cleanExp.endsWith('?')) {
          cleanExp += '.';
        }
        
        return cleanExp;
      })
      .filter(exp => exp !== null);
    
    if (cleanExplanations.length > 0) {
      
      details.innerHTML += `<div class="explanation-container" style="background-color: #f9f9f9; border-radius: 6px; padding: 8px; margin: 5px 0;">`;
      
      
      const formattedExplanation = cleanExplanations.join(' ');
      details.innerHTML += `<p style="margin: 5px 0;">${formattedExplanation}</p>`;
      
      details.innerHTML += `</div>`;
    } else {
      
      if (result.prediction === 'non_covid') {
        details.innerHTML += `<div class="explanation-container" style="background-color: #f9f9f9; border-radius: 6px; padding: 8px; margin: 5px 0;">
          <p style="margin: 5px 0;">The text does not make any claims related to COVID-19 misinformation.</p>
        </div>`;
      } else if (result.prediction === 'fake') {
        details.innerHTML += `<div class="explanation-container" style="background-color: #f9f9f9; border-radius: 6px; padding: 8px; margin: 5px 0;">
          <p style="margin: 5px 0;">The content contains potential misinformation related to COVID-19.</p>
        </div>`;
      } else if (result.isOpinion) {
        details.innerHTML += `<div class="explanation-container" style="background-color: #f9f9f9; border-radius: 6px; padding: 8px; margin: 5px 0;">
          <p style="margin: 5px 0;">The content expresses personal opinions rather than verifiable facts.</p>
        </div>`;
      } else {
        details.innerHTML += `<div class="explanation-container" style="background-color: #f9f9f9; border-radius: 6px; padding: 8px; margin: 5px 0;">
          <p style="margin: 5px 0;">The content appears to contain factual information about COVID-19.</p>
        </div>`;
      }
    }
  }
  
  
  details.innerHTML += `<div style="font-style: italic; margin-top: 8px; font-size: 12px; padding: 5px; background-color: rgba(0,0,0,0.05); border-radius: 4px;">
    This is an automated assessment. Please verify information from reliable sources.
  </div>`;
  
  badge.appendChild(details);
  
  
  const tweetTextElement = tweetElement.querySelector('[data-testid="tweetText"]');
  if (tweetTextElement) {
    tweetTextElement.parentNode.insertBefore(badge, tweetTextElement.nextSibling);
  }
}


function getConfidenceColor(confidence) {
  if (confidence >= 70) return '#1da1f2'; 
  if (confidence >= 40) return '#ffd166'; 
  return '#ff6b6b'; 
}


function extractGeneralTopics(text) {
  const topics = [];
  const lowerText = text.toLowerCase();
  
  
  if (lowerText.match(/\b(politic|election|vote|democrat|republican|congress)\b/i)) {
    topics.push('Politics');
  }
  if (lowerText.match(/\b(econom|market|stock|inflation|recession|gdp)\b/i)) {
    topics.push('Economy');
  }
  if (lowerText.match(/\b(tech|ai|artificial intelligence|computer|software|app|digital)\b/i)) {
    topics.push('Technology');
  }
  if (lowerText.match(/\b(health|medical|doctor|hospital|disease|treatment)\b/i) && 
      !lowerText.match(/\b(covid|coronavirus)\b/i)) {
    topics.push('Health (non-COVID)');
  }
  if (lowerText.match(/\b(climate|environment|global warming|pollution|renewable|sustainable)\b/i)) {
    topics.push('Environment');
  }
  if (lowerText.match(/\b(sport|game|team|player|win|championship|tournament)\b/i)) {
    topics.push('Sports');
  }
  if (lowerText.match(/\b(entertainment|movie|film|tv|television|show|actor|actress|celebrity)\b/i)) {
    topics.push('Entertainment');
  }
  
  
  if (topics.length === 0) {
    topics.push('General content');
  }
  
  return topics;
}


function extractMisinfoIndicators(text) {
  const indicators = [];
  const lowerText = text.toLowerCase();
  
  
  if (lowerText.match(/\b(cure|cures|treatment|treats|prevents|heals|remedy)\b/i)) {
    indicators.push('Unverified cure or treatment claims');
  }
  if (lowerText.match(/\b(conspiracy|coverup|cover up|hoax|fake|scam|lie|lying)\b/i)) {
    indicators.push('Conspiracy terminology');
  }
  if (lowerText.match(/\b(they don't want you to know|what they won't tell you|secret|hidden truth)\b/i)) {
    indicators.push('Appeal to hidden knowledge');
  }
  if (lowerText.match(/\b(100\%|always|never|every|all|proven|guaranteed)\b/i)) {
    indicators.push('Absolute or exaggerated claims');
  }
  if (lowerText.match(/\b(study proves|research shows|scientists discovered)\b/i) && 
      !lowerText.match(/\b(doi|journal|publication|published in|link to study)\b/i)) {
    indicators.push('Unlinked scientific claims');
  }
  
  return indicators;
}


function updateAnalysisStats(result) {
  chrome.storage.local.get(['tweetsAnalyzed', 'misinfoDetected', 'opinionsDetected'], function(data) {
    const tweetsAnalyzed = (data.tweetsAnalyzed || 0) + 1;
    const misinfoDetected = (data.misinfoDetected || 0) + (result.prediction === 'fake' ? 1 : 0);
    const opinionsDetected = (data.opinionsDetected || 0) + (result.isOpinion ? 1 : 0);
    
    chrome.storage.local.set({
      tweetsAnalyzed: tweetsAnalyzed,
      misinfoDetected: misinfoDetected,
      opinionsDetected: opinionsDetected
    });
  });
}

const CACHE_SIZE_LIMIT = 100;
const analysisCache = new Map();

function addToCache(key, value) {
    if (analysisCache.size >= CACHE_SIZE_LIMIT) {
        const oldestKey = analysisCache.keys().next().value;
        analysisCache.delete(oldestKey);
    }
    
    analysisCache.set(key, value);
}

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

document.addEventListener('click', function(event) {
    const tweetElement = event.target.closest('[data-testid="tweet"]');
    
    if (!tweetElement) return;
    
    const tweetTextElement = tweetElement.querySelector('[data-testid="tweetText"]');
    if (!tweetTextElement) return;
    
    const authorElement = tweetElement.querySelector('[data-testid="User-Name"]');
    if (!authorElement) return;
    
    const tweetText = tweetTextElement.innerText;
    const tweetId = tweetElement.getAttribute('data-tweet-id') || tweetText.substring(0, 50);
    
    if (analysisCache.has(tweetId)) {
        console.log("Using cached analysis");
        const result = analysisCache.get(tweetId);
        displayClassificationResult(tweetText, result, tweetTextElement);
        return;
    }
    
    const authorInfo = {
        name: authorElement.innerText,
        verified: tweetElement.querySelector('[data-testid="icon-verified"]') !== null,
        handle: authorElement.querySelector('span')?.innerText || ''
    };
    
    console.log("Analyzing tweet:", tweetText.substring(0, 50) + "...");
    
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
        displayCredibleSourceBadge(tweetText, authorInfo, tweetTextElement);
        return;
    }
    
    showLoadingIndicator(tweetTextElement);
    
    setTimeout(() => {
        fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                text: tweetText
            }),
        })
        .then(response => {
            console.log("API Response Status:", response.status);
            return response.json();
        })
        .then(result => {
            analysisCache.set(tweetId, result);
            
            const features = {
                word_count: tweetText.split(/\s+/).filter(word => word.length > 0).length,
                hashtag_count: (tweetText.match(/#\w+/g) || []).length,
                mention_count: (tweetText.match(/@\w+/g) || []).length,
                url_count: (tweetText.match(/https?:\/\/[^\s]+/g) || []).length
            };
            
            result.features = features;
            
            
            const opinionResult = detectOpinion(tweetText);
            result.isOpinion = opinionResult.isOpinion;
            result.opinionWords = opinionResult.opinionWords;
            
            
            const tweetId = tweetElement.getAttribute('data-tweet-id') || tweetText.substring(0, 50);
            analysisCache.set(tweetId, result);
            
            
            displayClassificationResult(tweetText, result, tweetElement);
        })
        .catch(error => {
            console.error("API Error:", error);
            removeLoadingIndicator();
            displayErrorMessage(tweetTextElement);
        });
    }, 10);
});

function showLoadingIndicator(tweetElement) {
    const existingBadges = tweetElement.querySelectorAll('.misinfo-badge, .loading-indicator');
    existingBadges.forEach(badge => badge.remove());
    
    const loader = document.createElement('div');
    loader.className = 'loading-indicator';
    loader.innerHTML = `
        <div style="padding: 10px; margin-top: 10px; border-radius: 8px; background-color: #f5f8fa; text-align: center; font-family: Arial, sans-serif;">
            <div>Analyzing tweet...</div>
            <div style="margin-top: 5px; height: 4px; width: 100%; background-color: #e1e8ed; border-radius: 2px; overflow: hidden;">
                <div class="loading-bar" style="height: 100%; width: 30%; background-color: #1da1f2; animation: loading 1.5s infinite ease-in-out;"></div>
            </div>
        </div>
    `;
    
    if (!document.getElementById('loading-animation-style')) {
        const style = document.createElement('style');
        style.id = 'loading-animation-style';
        style.textContent = `
            @keyframes loading {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(400%); }
            }
        `;
        document.head.appendChild(style);
    }
    
    const tweetTextElement = tweetElement.querySelector('[data-testid="tweetText"]');
    if (tweetTextElement) {
        tweetTextElement.parentNode.insertBefore(loader, tweetTextElement.nextSibling);
    }
}

function removeLoadingIndicator() {
    const loadingIndicators = document.querySelectorAll('.loading-indicator');
    loadingIndicators.forEach(indicator => indicator.remove());
}

function displayErrorMessage(tweetElement) {
    const errorMessage = document.createElement('div');
    errorMessage.className = 'misinfo-badge';
    errorMessage.style.backgroundColor = '#f8d7da';
    errorMessage.style.color = '#721c24';
    errorMessage.style.padding = '10px';
    errorMessage.style.marginTop = '10px';
    errorMessage.style.borderRadius = '8px';
    errorMessage.style.fontFamily = 'Arial, sans-serif';
    errorMessage.innerHTML = `
        <strong>Error analyzing tweet</strong>
        <p>Could not connect to analysis server. Please make sure the server is running.</p>
    `;
    
    const tweetTextElement = tweetElement.querySelector('[data-testid="tweetText"]');
    if (tweetTextElement) {
        tweetTextElement.parentNode.insertBefore(errorMessage, tweetTextElement.nextSibling);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('Tweet Misinformation Detector Extension loaded');
    
    const style = document.createElement('style');
    style.id = 'loading-animation-style';
    style.textContent = `
        @keyframes loading {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(400%); }
        }
    `;
    document.head.appendChild(style);
});

const observer = new MutationObserver(debounce(function(mutations) {
    if (document.querySelector('[data-testid="primaryColumn"]')) {
        setupTweetAnalysis();
        observer.disconnect();
    }
}, 500));

observer.observe(document.body, { childList: true, subtree: true });

if (document.querySelector('[data-testid="primaryColumn"]')) {
    setupTweetAnalysis();
}

function setupTweetAnalysis() {
    const tweetSelector = 'article[data-testid="tweet"]';
    

    
   
    document.removeEventListener('click', tweetClickHandler);
    
    
    document.addEventListener('click', tweetClickHandler);
    
    function tweetClickHandler(event) {
        const tweetElement = event.target.closest(tweetSelector);
        if (!tweetElement) return;
        
        if (tweetElement.dataset.analyzed === 'true') return;
        
        tweetElement.dataset.analyzed = 'true';
        
        const tweetTextElement = tweetElement.querySelector('[data-testid="tweetText"]');
        if (!tweetTextElement) return;
        
        const tweetText = tweetTextElement.textContent;
        
        showLoadingIndicator(tweetElement);
        
        const authorNode = tweetElement.querySelector('[data-testid="User-Name"]');
        if (!authorNode) return;
        
        const authorInfo = {
            name: authorNode.innerText,
            verified: tweetElement.querySelector('[data-testid="icon-verified"]') !== null,
            handle: authorNode.querySelector('span') ? authorNode.querySelector('span').innerText : ''
        };
        
        console.log("Author Info:", authorInfo);
        
        
        const credibleSources = [
            {handle: "@US_FDA", name: "U.S. FDA"},
            {handle: "@CDCgov", name: "CDC"},
            {handle: "@WHO", name: "WHO"},
            {handle: "@NIH", name: "NIH"},
            {handle: "@HHSGov", name: "HHS.gov"}
        ];
        
        const isCredibleSource = credibleSources.some(source => 
            (authorInfo.handle && authorInfo.handle.includes(source.handle)) || 
            (authorInfo.name && authorInfo.name.includes(source.name))
        );
        
        console.log("Is credible source:", isCredibleSource);
        
        if (isCredibleSource && authorInfo.verified) {
            
            const credibleResult = {
                prediction: 'real',
                confidence: 0.95, 
                explanation: [`Content from verified official health authority: ${authorInfo.name}`],
                isOpinion: false,
                features: {
                    word_count: tweetText.split(/\s+/).length,
                    hashtag_count: (tweetText.match(/#\w+/g) || []).length,
                    mention_count: (tweetText.match(/@\w+/g) || []).length,
                    url_count: (tweetText.match(/https?:\/\/[^\s]+/g) || []).length
                }
            };
            
            displayClassificationResult(tweetText, credibleResult, tweetElement);
            return;
        }
        
        
        fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                text: tweetText
            }),
            
            signal: AbortSignal.timeout(5000)
        })
        .then(response => {
            console.log("API Response Status:", response.status);
            if (!response.ok) {
                throw new Error(`API request failed with status ${response.status}`);
            }
            return response.json();
        })
        .then(result => {
            console.log("API Result:", result);
            
            
            const features = {
                word_count: tweetText.split(/\s+/).filter(word => word.length > 0).length,
                hashtag_count: (tweetText.match(/#\w+/g) || []).length,
                mention_count: (tweetText.match(/@\w+/g) || []).length,
                url_count: (tweetText.match(/https?:\/\/[^\s]+/g) || []).length
            };
            
            result.features = features;
            
            
            const opinionResult = detectOpinion(tweetText);
            result.isOpinion = opinionResult.isOpinion;
            result.opinionWords = opinionResult.opinionWords;
            
            
            const tweetId = tweetElement.getAttribute('data-tweet-id') || tweetText.substring(0, 50);
            analysisCache.set(tweetId, result);
            
            
            displayClassificationResult(tweetText, result, tweetElement);
        })
        .catch(error => {
            console.error("API Error:", error);
            
            
            const fallbackResult = {
                prediction: 'unknown',
                confidence: 0,
                explanation: ['Analysis server unavailable. Please check your connection.'],
                isOpinion: false,
                features: {
                    word_count: tweetText.split(/\s+/).filter(word => word.length > 0).length,
                    hashtag_count: (tweetText.match(/#\w+/g) || []).length,
                    mention_count: (tweetText.match(/@\w+/g) || []).length,
                    url_count: (tweetText.match(/https?:\/\/[^\s]+/g) || []).length
                }
            };
            
            displayClassificationResult(tweetText, fallbackResult, tweetElement);
        });
    }
}

function extractAdvancedFeatures(text) {
  
  const word_count = text.split(/\s+/).filter(word => word.length > 0).length;
  const hashtag_count = (text.match(/#\w+/g) || []).length;
  const mention_count = (text.match(/@\w+/g) || []).length;
  const url_count = (text.match(/https?:\/\/[^\s]+/g) || []).length;
  
  
  const words = text.split(/\s+/).filter(word => word.length > 0);
  const capitalized_words = words.filter(word => word === word.toUpperCase() && word.length > 1).length;
  
  
  const exclamation_count = (text.match(/!/g) || []).length;
  const question_count = (text.match(/\?/g) || []).length;
  
  
  const has_numbers = /\d+/.test(text);
  const has_percentages = /\d+\s*%/.test(text);
  const has_dates = /\b\d{1,2}\/\d{1,2}\/\d{2,4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b/.test(text);
  
  
  const positive_words = countWords(text, [
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'perfect',
    'happy', 'positive', 'proven', 'effective', 'safe', 'beneficial'
  ]);
  
  const negative_words = countWords(text, [
    'bad', 'terrible', 'awful', 'worst', 'hate', 'dangerous', 'harmful', 'toxic',
    'deadly', 'negative', 'unsafe', 'risk', 'threat', 'scary', 'fear'
  ]);
  
  
  const conspiracy_terms = countWords(text.toLowerCase(), [
    'conspiracy', 'hoax', 'coverup', 'cover-up', 'truth', 'exposed', 'secret', 'hidden',
    'they don\'t want you to know', 'wake up', 'sheeple', 'plandemic', 'deep state',
    'new world order', 'illuminati', 'microchip', '5g', 'bill gates', 'mind control'
  ]);
  
  return {
    word_count,
    hashtag_count,
    mention_count,
    url_count,
    capitalized_words,
    exclamation_count,
    question_count,
    has_numbers,
    has_percentages,
    has_dates,
    positive_words,
    negative_words,
    conspiracy_terms
  };
}

function countWords(text, wordList) {
  const lowerText = text.toLowerCase();
  return wordList.reduce((count, word) => {
    return count + (lowerText.includes(word) ? 1 : 0);
  }, 0);
}

function analyzeSourceCredibility(authorInfo) {

  const credibleSources = [
    {handle: "@US_FDA", name: "U.S. FDA", score: 0.95},
    {handle: "@CDCgov", name: "CDC", score: 0.95},
    {handle: "@WHO", name: "WHO", score: 0.95},
    {handle: "@NIH", name: "NIH", score: 0.95},
    {handle: "@HHSGov", name: "HHS.gov", score: 0.95},
    {handle: "@NIAID", name: "NIAID", score: 0.95},
    {handle: "@CDCemergency", name: "CDC Emergency", score: 0.95},
    {handle: "@Surgeon_General", name: "U.S. Surgeon General", score: 0.95},
    {handle: "@PHEgov", name: "Public Health England", score: 0.9},
    {handle: "@ECDC_EU", name: "ECDC", score: 0.9},
    {handle: "@RedCross", name: "Red Cross", score: 0.85},
    {handle: "@MSF", name: "Doctors Without Borders", score: 0.85},
    {handle: "@UNICEF", name: "UNICEF", score: 0.85},
    {handle: "@UNAIDS", name: "UNAIDS", score: 0.85},
    {handle: "@unfpa", name: "UNFPA", score: 0.85}
  ];
  
  
  const newsOrganizations = [
    {handle: "@Reuters", name: "Reuters", score: 0.8},
    {handle: "@AP", name: "Associated Press", score: 0.8},
    {handle: "@BBCNews", name: "BBC News", score: 0.75},
    {handle: "@NPR", name: "NPR", score: 0.75},
    {handle: "@nytimes", name: "New York Times", score: 0.7},
    {handle: "@washingtonpost", name: "Washington Post", score: 0.7},
    {handle: "@FactCheck", name: "FactCheck.org", score: 0.85},
    {handle: "@snopes", name: "Snopes", score: 0.8},
    {handle: "@PolitiFact", name: "PolitiFact", score: 0.8}
  ];
  
  
  const academicSources = [
    {handle: "@JohnsHopkins", name: "Johns Hopkins", score: 0.9},
    {handle: "@HarvardHealth", name: "Harvard Health", score: 0.9},
    {handle: "@MayoClinic", name: "Mayo Clinic", score: 0.9},
    {handle: "@StanfordMed", name: "Stanford Medicine", score: 0.9},
    {handle: "@MountSinaiNYC", name: "Mount Sinai", score: 0.85}
  ];
  
  
  for (const source of credibleSources) {
    if (authorInfo.handle.includes(source.handle) || authorInfo.name.includes(source.name)) {
      return {
        isCredible: true,
        credibilityScore: authorInfo.verified ? source.score : source.score * 0.8,
        sourceType: 'Health Authority',
        sourceName: source.name
      };
    }
  }
  
  
  for (const org of newsOrganizations) {
    if (authorInfo.handle.includes(org.handle) || authorInfo.name.includes(org.name)) {
      return {
        isCredible: true,
        credibilityScore: authorInfo.verified ? org.score : org.score * 0.7,
        sourceType: 'News Organization',
        sourceName: org.name
      };
    }
  }
  
  
  for (const academic of academicSources) {
    if (authorInfo.handle.includes(academic.handle) || authorInfo.name.includes(academic.name)) {
      return {
        isCredible: true,
        credibilityScore: authorInfo.verified ? academic.score : academic.score * 0.75,
        sourceType: 'Academic Institution',
        sourceName: academic.name
      };
    }
  }
  
  
  return {
    isCredible: false,
    credibilityScore: authorInfo.verified ? 0.5 : 0.25,
    sourceType: 'Unknown',
    sourceName: authorInfo.name
  };
}

function calculateMisinformationScore(text, authorInfo, apiResult) {
  
  const sourceAnalysis = analyzeSourceCredibility(authorInfo);
  
  
  const opinionAnalysis = detectOpinion(text);
  
  
  const features = extractAdvancedFeatures(text);
  
  
  const riskFactors = {
    excessive_caps: features.capitalized_words > 3 ? 0.12 : 0,
    excessive_punctuation: (features.exclamation_count > 3) ? 0.08 : 0,
    very_short_message: features.word_count < 10 ? 0.07 : 0,
    highly_emotional: (features.positive_words + features.negative_words) > 5 ? 0.1 : 0,
    lacks_evidence: (!features.has_numbers && !features.has_percentages && !features.has_dates) ? 0.15 : 0,
    contains_conspiracy_terms: features.conspiracy_terms > 0 ? 0.2 : 0,
    lacks_sources: features.url_count === 0 ? 0.1 : 0
  };
  
  
  const riskScore = Object.values(riskFactors).reduce((sum, score) => sum + score, 0);
  
 
  let finalScore;
  
  if (apiResult && apiResult.confidence) {
    
    finalScore = apiResult.prediction === 'fake' ? 
      0.65 + (apiResult.confidence * 0.35) - (sourceAnalysis.credibilityScore * 0.6) + riskScore :
      0.25 - (apiResult.confidence * 0.35) - (sourceAnalysis.credibilityScore * 0.6) + riskScore;
  } else {
    
    finalScore = 0.5 - (sourceAnalysis.credibilityScore * 0.6) + riskScore + (opinionAnalysis.isOpinion ? 0.15 : 0);
  }
  
 
  finalScore = Math.max(0, Math.min(1, finalScore));
  
  return {
    misinformationScore: finalScore,
    confidence: 1 - (finalScore * 0.5), 
    isLikelyMisinformation: finalScore > 0.6,
    isOpinion: opinionAnalysis.isOpinion,
    sourceCredibility: sourceAnalysis,
    riskFactors: riskFactors
  };
}