import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SideNavIconsComponent } from './side-nav-icons.component';

describe('SideNavIconsComponent', () => {
  let component: SideNavIconsComponent;
  let fixture: ComponentFixture<SideNavIconsComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ SideNavIconsComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(SideNavIconsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
