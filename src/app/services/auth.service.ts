import { Injectable } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/auth';
import { Router } from '@angular/router';
import { MatSnackBar } from '@angular/material';
import * as firebase from 'firebase';
import { Observable, from, of, Subscription } from 'rxjs';
import { AngularFirestoreDocument, AngularFirestore } from '@angular/fire/firestore';
import { switchMap } from 'rxjs/operators';
import { User } from '../interfaces/user';

@Injectable({
  providedIn: 'root'
})
export class AuthService {

  public currentUser;
  defaultImage: string;
  subscription: Subscription[] = [];

  usersRef: firebase.firestore.CollectionReference = this.db.collection('users').ref;

  constructor(private afauth: AngularFireAuth, private route: Router, private snackBar: MatSnackBar, private db: AngularFirestore) {

    this.currentUser = this.afauth.authState.pipe(
      switchMap(user => {
        if (user) {
          return this.db.doc(`users/${user.uid}`).valueChanges();
        } else {
          return of(null);
        }
      })
    );
  }



  signup(name: string, email: string, password: string) {
    return from(
      this.afauth.auth
        .createUserWithEmailAndPassword(email, password)
        .then(data => {
          console.log(data);
          return this.updateUserDetailsOnSignup(data, name);
        })
        .catch(err => err)
    );
  }

  signupWithGmail() {
    return from(
      this.afauth.auth.signInWithPopup(new firebase.auth.GoogleAuthProvider())
        .then((data) => {
          if (data.additionalUserInfo.isNewUser) {
            console.log('new user', data);
            return this.updateUserDetailsOnSignup(data, data.user.displayName);
          } else {
            return this.updateUserDetailsOnLogin(data);
          }
        }).catch((err) => {
          return err;
        })
    );
  }

  signupWithGithub() {
    return from(
      this.afauth.auth.signInWithPopup(new firebase.auth.GithubAuthProvider())
        .then((data) => {
          console.log(data, 'github');
          if (data.additionalUserInfo.isNewUser) {
            return this.updateUserDetailsOnSignup(data, data.additionalUserInfo.username);
          } else {
            return this.updateUserDetailsOnLogin(data);
          }
        }).catch((err) => {
          return err;
        })
    );
  }

  login(email: string, password: string) {
    return from(
      this.afauth.auth.signInWithEmailAndPassword(email, password)
        .then(data => {
          return this.updateUserDetailsOnLogin(data);
        }).catch(err => {
          return err;
        })
    );
  }

  loginWithGmail() {
    return from(
      this.afauth.auth.signInWithPopup(new firebase.auth.GoogleAuthProvider())
        .then((data) => {
          console.log(data);
          if (!data.additionalUserInfo.isNewUser) {
            return this.updateUserDetailsOnLogin(data);
          } else {
            return this.updateUserDetailsOnSignup(data, data.user.displayName);
          }
        }).catch((err) => {
          return err;
        })
    );
  }

  loginWithGithub() {
    return from(
      this.afauth.auth.signInWithPopup(new firebase.auth.GithubAuthProvider())
        .then((data) => {
          if (!data.additionalUserInfo.isNewUser) {
            return this.updateUserDetailsOnLogin(data);
          } else {
            return this.updateUserDetailsOnSignup(data, data.user.displayName);
          }
        }).catch((err) => {
          return err;
        })
    );
  }

  passwordReset(email: string) {
    return from(
      this.afauth.auth.sendPasswordResetEmail(email)
        .then(() => {
          return 'success';
        })
        .catch(err => {
          return err;
        })
    );
  }

  signOut() {
    return from(this.afauth.auth.signOut().then(() => true).catch(() => false));
  }
  
  updateUserDetailsOnSignup(data, name) {
    console.log(data);
    const userRef: AngularFirestoreDocument = this.db.doc(`users/${data.user.uid}`);

    const updatedUser: User = {
      id: data.user.uid,
      name: data.user.displayName || name,
      email: data.user.email,
      photoURL: data.user.photoURL || 'https://firebasestorage.googleapis.com/v0/b/chatelectron-44eab.appspot.com/o/profile_boy.png?alt=media&token=2f3a5ada-4baa-4639-a740-c66f20a30729',
      isAdmin: false
    };
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

  currentUserDetails() {
    return this.afauth.auth.currentUser;
  }
}
