import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
@Injectable({
  providedIn: 'root'
})
export class CommonService {
  loggedIn = new BehaviorSubject<boolean>(false);
  constructor() { }
}
