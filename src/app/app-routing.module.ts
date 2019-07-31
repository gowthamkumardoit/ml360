import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { HomeComponent } from './home/home.component';
import { PreviewComponent } from './preview/preview.component';
import { FeatureSelectionComponent } from './feature-selection/feature-selection.component';
import { ResultComponent } from './result/result.component';
import { LoginComponent } from './login/login.component';

import { AuthGuard } from './auth/auth.guard';

const appRoutes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  { path: 'login', component: LoginComponent },
  { path: 'home', component: HomeComponent, },
  { path: 'preview', component: PreviewComponent, },
  { path: 'feature-selection', component: FeatureSelectionComponent, },
  { path: 'result', component: ResultComponent, },
];

@NgModule({
  imports: [RouterModule.forRoot(appRoutes, { onSameUrlNavigation: 'reload' })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
