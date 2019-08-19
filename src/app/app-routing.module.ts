import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { HomeComponent } from './components/home/home.component';
import { PreviewComponent } from './components/preview/preview.component';
import { FeatureSelectionComponent } from './components/feature-selection/feature-selection.component';
import { ResultComponent } from './components/result/result.component';
import { LoginComponent } from './components/login/login.component';
import { SignupComponent } from './components/signup/signup.component';

import { AuthGuard } from './auth/auth.guard';
import { ResetPasswordComponent } from './components/reset-password/reset-password.component';
import { SecureInnerRoutesGuard } from './auth/secure-inner-routes.guard';
import { SettingsComponent } from './components/settings/settings.component';
import { HistoryComponent } from './components/history/history.component';
import { PreviewResolver } from './resolvers/preview.resolve';

const appRoutes: Routes = [
  { path: '', redirectTo: 'signup', pathMatch: 'full' },
  { path: 'signup', component: SignupComponent, },
  { path: 'login', component: LoginComponent, },
  { path: 'passwordReset', component: ResetPasswordComponent, },
  { path: 'home', component: HomeComponent, canActivate: [AuthGuard],  },
  { path: 'preview', component: PreviewComponent, canActivate: [AuthGuard], resolve: {response: PreviewResolver} },
  { path: 'feature-selection', component: FeatureSelectionComponent, canActivate: [AuthGuard] },
  { path: 'result', component: ResultComponent, canActivate: [AuthGuard] },
  { path: 'settings', component: SettingsComponent, canActivate: [AuthGuard] },
  { path: 'history', component: HistoryComponent, canActivate: [AuthGuard] },
];

@NgModule({
  imports: [RouterModule.forRoot(appRoutes, { onSameUrlNavigation: 'reload' })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
