'use strict';

import { 
    TitledPage,
    TabsView,
    MyDataTable,
    MySessionsTable,
    ClientLocation,
    Button,
    Api,
    Tools
} from '@src/classes';

export class MePage extends TitledPage {

    private tabsView: TabsView;

    constructor() {

        super('My account', 'my-account');
        
        const logoutButton = new Button({
            label: 'Log out',
            class: 'small'
        }, this.actions).onNative('click', async (event: Event) => {

            event.stopPropagation();

            logoutButton.load();

            try {
                await Api.post('/logout');
                ClientLocation.get().api.checkAuth();
                await Tools.sleep(500);
                logoutButton.unload();
            } catch(error) {
                logoutButton.unload();
                ClientLocation.get().api.checkAuth();
            }
        });

        this.tabsView = new TabsView([
            {
                text: 'Data',
                event: 'data'
            },
            {
                text: 'Sessions',
                event: 'sessions'
            }
        ], this.content);

        this.tabsView.addClass('light-zone');

        this.tabsView.on('data', () => {
            new MyDataTable(this.tabsView.view.empty());
        });

        this.tabsView.on('sessions', () => {
            new MySessionsTable(this.tabsView.view.empty());
        });
    }
}