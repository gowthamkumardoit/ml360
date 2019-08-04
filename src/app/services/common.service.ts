import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { MatSnackBar } from '@angular/material';
@Injectable({
  providedIn: 'root'
})
export class CommonService {
  loggedIn = new BehaviorSubject<boolean>(false);
  constructor(private snackBar: MatSnackBar) { }

}
