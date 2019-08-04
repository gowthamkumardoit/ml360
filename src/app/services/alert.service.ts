import { Injectable } from '@angular/core';
import { MatSnackBar } from '@angular/material';

@Injectable({
  providedIn: 'root'
})
export class AlertsService {

  constructor(private snackBar: MatSnackBar) { }

  setAlertForLogin(data) {
    if (data && data.code) {
      switch (data.code) {
          case 'auth/wrong-password': {
            this.snackBar.open('Wrong Password', 'Close', {duration: 3000});
            break;
          }
          case 'auth/invalid-email': {
            this.snackBar.open(data.message, 'Close', {duration: 3000});
            break;
          }
          case 'auth/user-not-found': {
            this.snackBar.open('No User Found', 'Close', {duration: 3000});
            break;
          }
      }
    }
  }

  setAlertsForAuthProviderLogins(data) {
    if (data && data.code) {
      switch (data.code) {
          case 'auth/account-exists-with-different-credential': {
            this.snackBar.open('Accounts exists with different credential', 'Close', {duration: 3000});
            break;
          }
          case 'auth/cancelled-popup-request': {
            this.snackBar.open('Login popup closed', 'Close', {duration: 3000});
            break;
          }
          case 'auth/operation-not-allowed': {
            this.snackBar.open('Operation not allowed / Account disabled.', 'Close', {duration: 3000});
            break;
          }
          case 'auth/popup-blocked': {
           this.snackBar.open(data.code, 'Close', {duration: 3000});
           break;
         }
         case 'auth/popup-closed-by-user': {
           this.snackBar.open('Popup has been closed. Please try again.', 'Close', {duration: 3000});
           break;
         }
      }
    }
  }

  showError(msg) {
    this.snackBar.open(msg, 'close', { duration: 2000 });
  }
}
