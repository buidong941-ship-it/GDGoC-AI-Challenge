const REGISTRATION_WEBHOOK_URL = "Your Production Registration Webhook URL";
const REGISTRATION_WEBHOOK_BEARER_TOKEN = "Your Registration Bearer Token";

// Organizer-managed content placeholders.
const SUBMISSION_FORM_LINK = "Your Submission Form Link";
const DISCORD_COMMUNITY_LINK = "Your Discord Community Link";
const CONTACT_HELP_CHANNEL = "devclub.hcmus@gmail.com";

function onFormSubmit(e) {
  // Works reliably for Form submit triggers
  const answers = {};
  const itemResponses = e.response.getItemResponses();

  itemResponses.forEach((ir) => {
    const title = String(ir.getItem().getTitle() || "").trim();
    const value = String(ir.getResponse() || "").trim();
    answers[title] = value;
  });

  Logger.log("Form titles received: " + JSON.stringify(Object.keys(answers)));

  // Map with tolerant fallback for Team Name/title variations
  const payload = {
    "Team Name": answers["Team Name"] || answers["Team name"] || "",
    "Primary contact name": answers["Primary contact name"] || "",
    "Primary contact email": answers["Primary contact email"] || "",
    "Second contact name": answers["Second contact name"] || "",
    "Second contact email": answers["Second contact email"] || "",
    "Agreement to rules": answers["Agreement to rules"] || "",
  };

  const response = UrlFetchApp.fetch(REGISTRATION_WEBHOOK_URL, {
    method: "post",
    contentType: "application/json",
    headers: {
      Authorization: "Bearer " + REGISTRATION_WEBHOOK_BEARER_TOKEN,
    },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true,
  });

  const statusCode = response.getResponseCode();
  const bodyText = response.getContentText() || "{}";
  const result = JSON.parse(bodyText);

  if (statusCode !== 200 || result.status !== "success") {
    Logger.log("Registration webhook failed: code=" + statusCode + " body=" + bodyText);
    return;
  }

  const email = payload["Primary contact email"];
  const teamName = result.team_name;
  const canonicalTeamId = result.canonical_team_id;
  const submissionToken = result.submission_token;

  const subject = "GDGoC AI Challenge 2026 - Registration Approved";
  const lines = [
    "Hello " + teamName + ",",
    "",
    "Your registration is approved.",
    "",
    "Team name: " + teamName,
    "Canonical team ID: " + canonicalTeamId,
    "Reusable submission token: " + submissionToken,
    "",
    "Submission form: " + SUBMISSION_FORM_LINK,
    "Discord/community link: " + DISCORD_COMMUNITY_LINK,
    "Contact/help channel: " + CONTACT_HELP_CHANNEL,
    "",
    "Submission constraints and format:",
    "- Upload exactly one .zip file.",
    "- The zip must contain exactly one agent.py.",
    "- No path traversal, no symlinks, no nested archives.",
    "",
    "Thank you for participating! We hope you have a fun and valuable competition experience. :')",
    // CUSTOM_EMAIL_CONTENT,
  ];

  MailApp.sendEmail({
    to: email,
    subject: subject,
    body: lines.join("\n"),
  });
}

function installOnSubmitTrigger() {
  const form = FormApp.getActiveForm();
  ScriptApp.newTrigger("onFormSubmit")
    .forForm(form)
    .onFormSubmit()
    .create();
}

function singleValue(valueArray) {
  if (!valueArray || valueArray.length === 0) {
    return "";
  }
  return String(valueArray[0]).trim();
}
