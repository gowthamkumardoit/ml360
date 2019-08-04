import { Injectable } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/auth';
import { Router } from '@angular/router';
import { MatSnackBar } from '@angular/material';
import * as firebase from 'firebase';
import { Observable, from, of, Subscription, BehaviorSubject } from 'rxjs';
import { AngularFirestoreDocument, AngularFirestore } from '@angular/fire/firestore';
import { switchMap } from 'rxjs/operators';
import { User } from '../interfaces/user';
import { AlertsService } from './alert.service';
import { FirebaseAuth } from '@angular/fire';
import { CommonService } from './common.service';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  userData: any;
  public currentUser;
  defaultImage: string;
  subscription: Subscription[] = [];
  isAuthenticated;
  usersRef: firebase.firestore.CollectionReference = this.db.collection('users').ref;
  userObject = new BehaviorSubject({});

  constructor(private afauth: AngularFireAuth, private route: Router, private snackBar: MatSnackBar, private db: AngularFirestore, private alertService: AlertsService,
    private commonService: CommonService) {

    this.currentUser = this.afauth.authState.pipe(
      switchMap(user => {
        if (user) {
          return this.db.doc(`users/${user.uid}`).valueChanges();
        } else {
          return of(null);
        }
      })
    );

    // Setting logged in user in localstorage else null
    this.afauth.authState.subscribe(user => {
      if (user) {
        this.userData = user;
        localStorage.setItem('user', JSON.stringify(this.userData));
        JSON.parse(localStorage.getItem('user'));
        this.userObject.next(user);
      } else {
        localStorage.setItem('user', null);
        JSON.parse(localStorage.getItem('user'));
        this.userObject.next({});
      }
    });

  }


  signup(name: string, email: string, password: string) {
    this.afauth.auth
      .createUserWithEmailAndPassword(email, password)
      .then(data => {
        this.updateUserDetailsOnSignup(data, name);
        if (data.user) {
          this.route.navigate(['/home']);
          this.snackBar.open('Congrats! Account Created Successfully!', 'Close', { duration: 3000 });
        }
      })
      .catch(err => {
        if (err && err.code) {
          this.snackBar.open(err.code, 'Close', { duration: 3000 });
        }
      });
  }

  signupWithGmail() {
    this.afauth.auth.signInWithPopup(new firebase.auth.GoogleAuthProvider())
      .then((data) => {
        if (data.additionalUserInfo.isNewUser) {
          this.snackBar.open('Congrats! Account Created Successfully!', 'Close', { duration: 3000 });
          this.updateUserDetailsOnSignup(data, data.user.displayName);
        } else {
          this.snackBar.open('Successfully Logged In!', 'Close', { duration: 3000 });
          this.updateUserDetailsOnLogin(data);
        }
        this.route.navigate(['/home']);
      }).catch((err) => {
        if (err && err.code) {
          this.alertService.setAlertsForAuthProviderLogins(err);
        }
      });
  }

  signupWithGithub() {
    this.afauth.auth.signInWithPopup(new firebase.auth.GithubAuthProvider())
      .then((data) => {
        if (data.additionalUserInfo.isNewUser) {
          this.snackBar.open('Congrats! Account Created Successfully!', 'Close', { duration: 3000 });
          this.updateUserDetailsOnSignup(data, data.additionalUserInfo.username);
        } else {
          this.snackBar.open('Successfully Logged In!', 'Close', { duration: 3000 });
          this.updateUserDetailsOnLogin(data);
        }
        this.route.navigate(['/home']);
      }).catch((err) => {
        if (err && err.code) {
          this.alertService.setAlertsForAuthProviderLogins(err);
        }
      });
  }

  login(email: string, password: string) {
    this.afauth.auth.signInWithEmailAndPassword(email, password)
      .then(data => {
        this.updateUserDetailsOnLogin(data);
        if (data.user) {
          this.route.navigate(['/home']);
          this.snackBar.open('Successfully Logged In!', 'Close', { duration: 3000 });
          this.commonService.getCurrentLoggedInUser();
        }
      }).catch(err => {
        if (err && err.code) {
          this.alertService.setAlertForLogin(err);
        }
      });
  }

  loginWithGmail() {
    this.afauth.auth.signInWithPopup(new firebase.auth.GoogleAuthProvider())
      .then((data) => {
        if (!data.additionalUserInfo.isNewUser) {
          this.snackBar.open('Successfully Logged In!', 'Close', { duration: 3000 });
          this.updateUserDetailsOnLogin(data);
        } else {
          this.snackBar.open('Congrats! Account Created Successfully!', 'Close', { duration: 3000 });
          this.updateUserDetailsOnSignup(data, data.user.displayName);
        }
        this.route.navigate(['/home']);
      }).catch((err) => {
        if (err && err.code) {
          this.alertService.setAlertsForAuthProviderLogins(err);
        }
      });
  }

  loginWithGithub() {
    this.afauth.auth.signInWithPopup(new firebase.auth.GithubAuthProvider())
      .then((data) => {
        if (!data.additionalUserInfo.isNewUser) {
          this.snackBar.open('Successfully Logged In!', 'Close', { duration: 3000 });
          this.updateUserDetailsOnLogin(data);
        } else {
          this.snackBar.open('Congrats! Account Created Successfully!', 'Close', { duration: 3000 });
          this.updateUserDetailsOnSignup(data, data.user.displayName);
        }
        this.route.navigate(['/home']);
      }).catch((err) => {
        if (err && err.code) {
          this.alertService.setAlertsForAuthProviderLogins(err);
        }
      });
  }

  passwordReset(email: string) {
    return from(
      this.afauth.auth.sendPasswordResetEmail(email)
        .then(() => {
          this.route.navigate(['/login']);
          return 'success';
        })
        .catch(err => {
          return err;
        })
    );
  }

  signOut() {

    return from(this.afauth.auth.signOut().then(() => {
      localStorage.removeItem('user');
      this.route.navigate(['login']);
    }).catch(() => false));
  }

  updateUserDetailsOnSignup(data, name) {
    console.log(name);
    const userRef: AngularFirestoreDocument = this.db.doc(`users/${data.user.uid}`);
    const updatedUser: User = {
      id: data.user.uid,
      name: data.user.displayName || name,
      email: data.user.email,
      photoURL: data.user.photoURL,
      isAdmin: false
    };
    console.log(updatedUser);
    userRef.set(updatedUser);
    return data;
  }

  updateUserDetailsOnLogin(data) {

    const userRef: AngularFirestoreDocument = this.db.doc(`users/${data.user.uid}`);
    const query = this.usersRef.where('email', '==', this.afauth.auth.currentUser.email);

    query.get().then(snapshot => {
      const obj = {
        ...snapshot.docs[0].data(),
      };
      userRef.set(obj);
    });

    return data;
  }



  // Returns true when user is looged in and email is verified
  get isLoggedIn(): boolean {
    const user = JSON.parse(localStorage.getItem('user'));
    return (user !== null) ? true : false;
  }
}
