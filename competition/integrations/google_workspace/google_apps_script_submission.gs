/**
 * Google Apps Script Trigger for Submission Form
 * 
 * This script should be deployed as the onFormSubmit trigger for the submission form.
 * It extracts form responses and calls the Flask /submit endpoint for validation and processing.
 * 
 * Configuration:
 * 1. Replace SUBMISSION_WEBHOOK_URL with your Flask deployment URL (e.g., https://your-domain.com/submit)
 * 2. Replace SUBMISSION_WEBHOOK_AUTH_TOKEN with the same Bearer token used in Flask (env var REGISTRATION_WEBHOOK_AUTH_TOKEN)
 */

const SUBMISSION_WEBHOOK_URL = "Your Production Submission Webhook URL";

const SUBMISSION_WEBHOOK_AUTH_TOKEN = "Your Production Submission Webhook Auth Token";  // TODO: Replace with your Bearer token

/**
 * Main trigger handler for form submissions
 * 
 * Form should have these fields (in order):
 * 1. Canonical Team ID (short text)
 * 2. Submission Token (short text)
 * 3. Changelog (paragraph text)
 * 4. Submission ZIP (file upload - Google Drive link)
 * 5. Notes (optional, paragraph text)
 */
function onFormSubmit(e) {
  try {
    const response = e.response;
    const itemResponses = response.getItemResponses();
    
    // Expected field order from form:
    // [0] = Canonical Team ID
    // [1] = Submission Token
    // [2] = Changelog
    // [3] = Submission ZIP (Drive link)
    // [4] = Notes (optional)
    
    if (itemResponses.length < 4) {
      Logger.log("Error: Form has fewer than 4 required fields");
      return;
    }
    
    const canonicalTeamId = normalizeFormResponse(itemResponses[0].getResponse());
    const submissionToken = normalizeFormResponse(itemResponses[1].getResponse());
    const changelog = normalizeFormResponse(itemResponses[2].getResponse());
    const submissionLink = normalizeFileUploadResponse(itemResponses[3].getResponse());
    
    // Validate required fields
    if (!canonicalTeamId || !submissionToken || !submissionLink) {
      Logger.log("Error: Missing required fields");
      return;
    }
    
    // Extract file ID from Google Drive share link
    // Example link: https://drive.google.com/file/d/1ABC...XYZ/view?usp=sharing
    const fileId = extractFileIdFromLink(submissionLink);
    if (!fileId) {
      Logger.log("Error: Could not extract file ID from link: " + submissionLink);
      return;
    }
    
    // Build webhook payload
    const payload = {
      canonical_team_id: canonicalTeamId,
      submission_token: submissionToken,
      drive_file_id: fileId,
      changelog: changelog,
      original_filename: null  // Will be retrieved from Drive API
    };
    
    // Call Flask endpoint
    const options = {
      method: 'post',
      headers: {
        'Authorization': 'Bearer ' + SUBMISSION_WEBHOOK_AUTH_TOKEN,
        'Content-Type': 'application/json'
      },
      payload: JSON.stringify(payload),
      muteHttpExceptions: true
    };
    
    const response_http = UrlFetchApp.fetch(SUBMISSION_WEBHOOK_URL, options);
    const status = response_http.getResponseCode();
    const responseText = response_http.getContentText();
    
    Logger.log(`Submission webhook response [${status}]: ${responseText}`);
    
    // Parse response to check success
    try {
      const result = JSON.parse(responseText);
      if (status === 200 && result.status === "success") {
        Logger.log(`✓ Submission accepted. ID: ${result.submission_id}, Remaining today: ${result.remaining_today}`);
        // Optionally send confirmation email to team
        // sendSubmissionConfirmationEmail(canonicalTeamId, result);
      } else {
        Logger.log(`✗ Submission rejected. Error: ${result.error}, Reason: ${result.reason}`);
      }
    } catch (e) {
      Logger.log("Warning: Could not parse webhook response as JSON");
    }
    
  } catch (e) {
    Logger.log("Error in onFormSubmit: " + e.toString());
  }
}

/**
 * Extract Google Drive file ID from various link formats
 * 
 * Handles:
 * - https://drive.google.com/file/d/{FILE_ID}/view
 * - https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing
 * - https://drive.google.com/open?id={FILE_ID}
 * - Just the FILE_ID itself
 * 
 * Returns:
 * - String file ID if found, null otherwise
 */
function extractFileIdFromLink(link) {
  if (!link) return null;
  
  // Format 1: /file/d/{FILE_ID}/view
  let match = link.match(/\/file\/d\/([^\/]+)/);
  if (match && match[1]) return match[1];
  
  // Format 2: ?id={FILE_ID}
  match = link.match(/[?&]id=([^&]+)/);
  if (match && match[1]) return match[1];
  
  // Format 3: Plain FILE_ID (if already extracted)
  if (link.match(/^[a-zA-Z0-9_-]+$/) && link.length > 20) {
    return link;
  }
  
  return null;
}

/**
 * Normalize a regular form response into a trimmed string.
 */
function normalizeFormResponse(value) {
  if (value === null || value === undefined) {
    return "";
  }
  if (Array.isArray(value)) {
    return value.length > 0 ? String(value[0]).trim() : "";
  }
  return String(value).trim();
}

/**
 * Normalize a file-upload response into a single string link or file id.
 *
 * Google Forms can return an array for file upload questions. Since this
 * form is configured for one file per row, we use the first uploaded file.
 */
function normalizeFileUploadResponse(value) {
  if (value === null || value === undefined) {
    return "";
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "";
    }
    if (value.length > 1) {
      Logger.log("Warning: multiple uploaded files detected; using the first file only.");
    }
    return String(value[0]).trim();
  }
  return String(value).trim();
}

/**
 * Optional: Send confirmation email to team after successful submission
 * 
 * Customize this function to send a nice confirmation email
 */
function sendSubmissionConfirmationEmail(canonicalTeamId, result) {
  // TODO: Implement if needed
  // Could send email via MailApp or Gmail API
}

/**
 * Install the trigger programmatically
 * 
 * Run this function once to install the onFormSubmit trigger:
 * - Go to Extensions → Apps Script
 * - Create a new function with this code
 * - Run installSubmissionTrigger() once
 * - Delete this function after running
 */
function installSubmissionTrigger() {
  const form = FormApp.getActiveForm();
  ScriptApp.newTrigger("onFormSubmit")
    .forForm(form)
    .onFormSubmit()
    .create();
  Logger.log("✓ Submission trigger installed successfully");
}

/**
 * Test function to verify webhook connectivity
 * 
 * Run this to test if the Flask endpoint is reachable
 */
function testWebhookConnectivity() {
  try {
    const testPayload = {
      canonical_team_id: "test_team_123",
      submission_token: "test_token_abc123",
      drive_file_id: "test_file_xyz",
      changelog: "Test submission",
      original_filename: null
    };
    
    const options = {
      method: 'post',
      headers: {
        'Authorization': 'Bearer ' + SUBMISSION_WEBHOOK_AUTH_TOKEN,
        'Content-Type': 'application/json'
      },
      payload: JSON.stringify(testPayload),
      muteHttpExceptions: true
    };
    
    const response = UrlFetchApp.fetch(SUBMISSION_WEBHOOK_URL, options);
    const status = response.getResponseCode();
    const text = response.getContentText();
    
    Logger.log(`Webhook test response [${status}]: ${text}`);
    if (status >= 200 && status < 500) {
      Logger.log("✓ Webhook connectivity test PASSED");
      if (status === 401) {
        Logger.log("Note: auth failed, but the endpoint is reachable.");
      } else if (status === 400) {
        Logger.log("Note: request reached Flask and was rejected as invalid, which still confirms connectivity.");
      } else if (status === 503) {
        Logger.log("Note: Flask is reachable, but Google Drive service was not initialized.");
      }
    } else {
      Logger.log("✗ Webhook connectivity test FAILED");
    }
  } catch (e) {
    Logger.log("✗ Webhook connectivity test FAILED: " + e.toString());
  }
}
