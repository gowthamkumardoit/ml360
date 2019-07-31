import { Injectable } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/auth';
import { Router } from '@angular/router';
import { MatSnackBar } from '@angular/material';
@Injectable({
  providedIn: 'root'
})
export class AuthService {
  authState = null;
  constructor(private afauth: AngularFireAuth, private route: Router, private snackBar: MatSnackBar) {

    this.afauth.authState.subscribe((state) => {
      this.authState = state;
    });
    
  }

  login(userCreds) {
    this.afauth.auth.signInWithEmailAndPassword(userCreds.email, userCreds.password).then((data) => {
      this.route.navigate(['/home']);
      this.snackBar.open('Logged In', 'close', { duration: 2000 });
    }).catch((err) => {
      this.snackBar.open(err.message, 'close', { duration: 2000 });
    });
    localStorage.setItem("user", this.afauth.auth.currentUser.uid);
  }

  logout() {
    this.afauth.auth.signOut();
    localStorage.removeItem("user");
    this.snackBar.open('Logged Out', 'close', { duration: 2000 });
    this.route.navigate(['/login']);

  }

  get authenticated():boolean {
    return this.authState !== null;
  }
}
