// Google Apps Script to save ratings to Google Sheets
// This script receives POST requests and saves them to a Google Sheet

function doPost(e) {
  try {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    const data = JSON.parse(e.postData.contents);

    // Prepare row data
    const timestamp = new Date(data.timestamp);
    const firstName = data.firstName || '';
    const lastName = data.lastName || '';
    const ratings = data.ratings;

    // Flatten ratings array - each audio set's ratings in separate columns
    const row = [timestamp, firstName, lastName];
    ratings.forEach((audioSet, setIdx) => {
      audioSet.forEach((rating, versionIdx) => {
        row.push(rating);
      });
    });

    // Append to sheet
    sheet.appendRow(row);

    return ContentService.createTextOutput(JSON.stringify({
      status: 'success'
    })).setMimeType(ContentService.MimeType.JSON);

  } catch (error) {
    return ContentService.createTextOutput(JSON.stringify({
      status: 'error',
      message: error.toString()
    })).setMimeType(ContentService.MimeType.JSON);
  }
}

// Initialize sheet with headers (run this once manually)
function setupSheet() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  // Create headers
  const headers = ['Timestamp', 'First Name', 'Last Name'];

  // 4 audio sets with 4 versions each (custom_steps10, original_steps10, custom_steps20, original_steps20)
  for (let set = 1; set <= 4; set++) {
    for (let version = 1; version <= 4; version++) {
      headers.push(`Audio${set}_V${version}`);
    }
  }

  sheet.clear();
  sheet.appendRow(headers);
  sheet.getRange(1, 1, 1, headers.length).setFontWeight('bold');
}
